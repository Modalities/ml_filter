import hashlib
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import click_pathlib

from ml_filter.analysis.aggregate_scores import aggregate_human_annotations, aggregate_scores_in_directory
from ml_filter.analysis.collect_ir_metrics import collect_ir_metrics
from ml_filter.analysis.evaluate_prompt_based_annotations import evaluate_prompt_based_annotations
from ml_filter.analysis.interrater_reliability import compute_interrater_reliability_metrics
from ml_filter.analysis.plot_score_distributions import plot_differences_in_scores, plot_scores
from ml_filter.compare_experiments import compare_experiments
from ml_filter.data_processing.deduplication import deduplicate_jsonl
# from ml_filter.inference_pipeline.run_pipeline import run_pipeline
from ml_filter.llm_client import LLMClient
from ml_filter.sample_from_hf_dataset import sample_from_hf_dataset, upload_file_to_hf
from ml_filter.training.annotator_model_pipeline import run_annotator_training_pipeline
from ml_filter.translate import TranslationServiceType, TranslatorFactory
from ml_filter.utils.chunk_data import chunk_jsonl
from ml_filter.utils.manipulate_datasets import apply_score_transforms, convert_hf_dataset_to_jsonl, split_dataset
from ml_filter.utils.manipulate_documents import merge_and_sort_jsonl_files
from ml_filter.utils.manipulate_prompt import add_target_language_to_prompt
from ml_filter.utils.statistics import compute_num_words_and_chars_in_jsonl, run_word_count_jsonl_files

input_file_path_option = click.option(
    "--input_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the input file.",
)
output_folder_path_option = click.option(
    "--output_folder_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the output directory of the translated files.",
)
translation_service_option = click.option(
    "--translation_service",
    type=click.Choice([service.value.upper() for service in TranslationServiceType], case_sensitive=False),
    required=True,
    help=f"Translator to use ({', '.join(service.value for service in TranslationServiceType)}).",
)
source_language_code_option = click.option(
    "--source_language_code",
    type=str,
    required=True,
    help="Language code of the source language.",
)

target_language_codes_option = click.option(
    "--target_language_codes",
    type=str,
    required=True,
    help="Comma-separated list of languages.",
)

aggregation_option = click.option(
    "--aggregation",
    type=str,
    required=True,
    help="""
        Specifies how scores for a document from the same file are aggregated. 
        Supported values:
        - "mean": Compute the average score.
        - "max": Use the maximum score.
        - "min": Use the minimum score.
        - "majority": Use the score that was voted the most. If there is a tie, take the average of the winners.
    """,
)

labels_option = click.option(
    "--labels",
    type=str,
    required=True,
    help="Comma-separated list of possible labels.",
)

batch_size_option = click.option(
    "--batch_size",
    type=int,
    default=100000,
    show_default=True,
    help="Number of documents to process in each batch.",
)

path_to_files_argument = click.argument("path_to_files", nargs=-1, type=click.Path(path_type=Path))


@click.group()
def main() -> None:
    pass


@main.command(name="score_documents")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to a file with the YAML config file.",
)
@click.option(
    "--experiment_id",
    type=str,
    required=False,
    help="Experiment id for the current job. (Used only in Slurm, format e.g., yyyy-mm-dd__hh-mm-ss/job_array_id)",
)
@click.option(
    "--rest_endpoint",
    type=str,
    required=True,
    help="The endpoint for the LLM service.",
)
def entry_point_score_documents(config_file_path: Path, rest_endpoint: str, experiment_id: Optional[str] = None):
    with open(config_file_path, "rb") as f:
        hash_value = hashlib.sha256(f.read()).hexdigest()[:8]
    experiment_id_postfix = datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + f"__{hash_value}"

    if experiment_id is None:
        experiment_id = experiment_id_postfix
    else:
        experiment_id = experiment_id + f"/{experiment_id_postfix}"
    llm_service = LLMClient(config_file_path=config_file_path, experiment_id=experiment_id, rest_endpoint=rest_endpoint)
    llm_service.run()


@main.command(name="compare_experiments")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to a file with the YAML config file.",
)
def entry_point_compare_experiments(config_file_path: Path):
    # TODO check if entry point still works. rename
    compare_experiments(config_file_path)


@main.command(name="annotator_training_pipeline")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to the config file.",
)
def entry_annotator_training_pipeline(config_file_path: Path):
    run_annotator_training_pipeline(config_file_path=config_file_path)


@main.command(name="chunk_jsonl")
@click.option(
    "--input_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the input JSONL file.",
)
@click.option(
    "--output_dir",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Directory where chunk files will be saved.",
)
@click.option(
    "--lines_per_chunk",
    type=int,
    required=True,
    help="Number of lines per chunk file.",
)
def chunk_jsonl_file(input_file_path: Path, output_dir: Path, lines_per_chunk: int):
    chunk_jsonl(input_file_path=input_file_path, output_dir=output_dir, lines_per_chunk=lines_per_chunk)


@main.command(name="add_target_language_to_prompt_yaml")
@click.option(
    "--input_file_path",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the input prompt (yaml).",
)
@click.option(
    "--output_dir",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Directory where chunk files will be saved.",
)
def add_target_language_to_prompt_yaml(input_file_path: Path, output_dir: Path):
    add_target_language_to_prompt(input_file_path=input_file_path, output_dir=output_dir)


@main.command(name="translate_flat_yaml")
@input_file_path_option
@output_folder_path_option
@click.option(
    "--ignore_tag_text",
    type=str,
    required=False,
    help=(
        "Tag name, e.g. notranslate, indicating which part of the translation "
        "should be ignored. The tag name is internally converted to the full tag, "
        "i.e. <notranslate> for the given example."
    ),
)
@source_language_code_option
@target_language_codes_option
@translation_service_option
def translate_flat_yaml_cli(
    input_file_path: Path,
    output_folder_path: Path,
    source_language_code: str,
    target_language_codes: str,
    translation_service: str,
    ignore_tag_text: Optional[str] = None,
):
    """
    CLI command to translate flat YAML files using either DeepL or OpenAI.
    """
    target_language_codes_list = _get_target_language_codes_list_helper(target_language_codes=target_language_codes)
    translator = _get_translator_helper(translation_service=translation_service)
    translator.translate_flat_yaml_to_multiple_languages(
        input_file_path=input_file_path,
        output_folder_path=output_folder_path,
        source_language_code=source_language_code,
        target_language_codes=target_language_codes_list,
    )


@main.command(name="interrater_reliability")
@path_to_files_argument
@click.option(
    "--output_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Write the computed metrics to this json-file.",
)
@aggregation_option
def interrater_reliability_cli(path_to_files: tuple[Path], output_file_path: Path, aggregation: Optional[str] = None):
    compute_interrater_reliability_metrics(
        path_to_files=path_to_files,
        output_file_path=output_file_path,
        aggregation=aggregation,
    )


@main.command(name="plot_scores")
@path_to_files_argument
@click.option("--output_dir", type=str, required=True)
@aggregation_option
@labels_option
def plot_scores_cli(path_to_files: tuple[Path], output_dir: str, aggregation: str, labels: list[str]) -> None:
    """Plot the differences in scores."""
    path_to_files = [Path(p) for p in path_to_files]
    plot_scores(
        path_to_files=path_to_files,
        output_dir=Path(output_dir),
        aggregation=aggregation,
        labels=[float(label) for label in labels.split(",")]
    )
    plot_differences_in_scores(
        path_to_files=path_to_files,
        output_dir=Path(output_dir),
        aggregation=aggregation,
        labels=[float(label) for label in labels.split(",")]
    )


@main.command(name="evaluate_prompt_based_annotations")
@click.option("--input_directory", type=click.Path(exists=True, path_type=Path))
@click.option("--output_directory", type=click.Path(exists=False, path_type=Path))
@click.option("--gt_data", type=click.Path(exists=True, path_type=Path))
@click.option("--aggregation", type=str, default="majority", help="Aggregation method for scores.")
@click.option("--labels", type=str, help="Comma-separated list of labels.")
def evaluate_prompt_based_annotations_cli(
    input_directory: Path,
    output_directory: Path,
    gt_data: Path,
    aggregation: str,
    labels: str,
) -> None:
    """CLI command to evaluate prompt-based annotations and compute inter-rater reliability metrics."""
    evaluate_prompt_based_annotations(
        input_directory=input_directory,
        output_directory=output_directory,
        gt_data=gt_data,
        aggregation=aggregation,
        labels=[float(label) for label in labels.split(",")],
    )


@main.command(name="aggregate_scores")
@click.argument("input_directory", type=click.Path(exists=True, path_type=Path))
@click.argument("output_directory", type=click.Path(exists=False, path_type=Path))
@labels_option
@aggregation_option
@batch_size_option
@click.option("--raw_data_lookup_dir", type=click.Path(exists=False, path_type=Path), required=False)
def evaluate_prompt_based_annotations_cli(
    input_directory: Path,
    output_directory: Path,
    aggregation: str,
    labels: str,
    batch_size: int,
    raw_data_lookup_dir: Optional[Path] = None,
) -> None:
    """CLI command to evaluate prompt-based annotations and compute inter-rater reliability metrics."""
    aggregate_scores_in_directory(
        input_directory=input_directory,
        output_directory=output_directory,
        aggregation=aggregation,
        labels=[float(l) for l in labels.split(",")],
        batch_size=batch_size,
        raw_data_lookup_dir=raw_data_lookup_dir,
    )
    
    
@main.command(name="aggregate_human_annotations")
@click.option(
    "--annotations_file_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the annotations file.",
)
@click.option(
    "--output_file_path",
    type=click.Path(exists=False, path_type=Path),
    required=True,
    help="Path to the output file.",
)
@click.option(
    "--raw_data_file_path",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to the raw data file.",
)
@labels_option
@aggregation_option
@batch_size_option
def aggregate_human_annotations_cli(
    annotations_file_path: Path,
    output_file_path: Path,
    raw_data_file_path: Path,
    labels: str,
    aggregation: str,
    batch_size: int,
) -> None:
    """
    CLI command to aggregate human annotations by comparing them to ground truth data.
    """
    aggregate_human_annotations(
        annotations_file_path=annotations_file_path,
        output_file_path=output_file_path,
        raw_data_file_path=raw_data_file_path,
        labels=[float(label) for label in labels.split(",")],
        aggregation=aggregation,
        batch_size=batch_size,
    )


@main.command(name="collect_ir_metrics")
@click.option("--input_directory", type=click.Path(exists=True, path_type=Path))
@click.option("--output_directory", type=click.Path(exists=False, path_type=Path))
@click.option(
    "--min_metrics",
    type=str,
    help="Comma-separated list of metrics for which lower is better."
    + "All other metrics are considered to be better when higher.",
)
def collect_ir_metrics_cli(input_directory: Path, output_directory: Path, min_metrics: str):
    """CLI command to evaluate prompt-based annotations and compute inter-rater reliability metrics."""
    collect_ir_metrics(
        input_directory=input_directory,
        output_directory=output_directory,
        min_metrics=[metric for metric in min_metrics.split(",")],
    )


@main.command(name="translate_jsonl_to_multiple_languages_cli")
@input_file_path_option
@output_folder_path_option
@source_language_code_option
@target_language_codes_option
@translation_service_option
def translate_jsonl_to_multiple_languages_cli(
    input_file_path: Path,
    output_folder_path: Path,
    source_language_code: str,
    target_language_codes: str,
    translation_service: str,
):
    target_language_codes_list = _get_target_language_codes_list_helper(target_language_codes=target_language_codes)
    translator = _get_translator_helper(translation_service=translation_service)
    translator.translate_jsonl_to_multiple_languages(
        input_file_path=input_file_path,
        output_folder_path=output_folder_path,
        source_language_code=source_language_code,
        target_language_codes=target_language_codes_list,
    )


@main.command(name="compute_num_words_in_jsonl_cli")
@input_file_path_option
@click.option(
    "--output_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to the output file.",
)
def compute_num_words_in_jsonl_cli(
    input_file_path: Path,
    output_file_path: Path,
):
    compute_num_words_and_chars_in_jsonl(input_file_path=input_file_path, output_file_path=output_file_path)


@main.command(name="sample_from_hf_dataset")
@click.option(
    "--dataset_name",
    required=True,
    type=str,
    help="Name of the Hugging Face dataset to sample from (e.g., 'HuggingFaceFW/fineweb-edu-llama3-annotations').",
)
@click.option(
    "--dataset_split",
    required=True,
    type=str,
    help="The split of the Hugging Face dataset that is used for sampling (e.g., 'train').",
)
@click.option(
    "--output_file_path",
    required=True,
    type=click.Path(),
    help="Path to save the sampled data as a JSON file (e.g., 'output.json').",
)
@click.option(
    "--column_name", required=True, type=str, help="Column in the dataset used for filtering (e.g., 'score')."
)
@click.option(
    "--column_values",
    required=True,
    type=str,
    help="Comma-separated list of relevant column values to sample.",
)
@click.option(
    "--num_docs_per_value",
    required=True,
    type=int,
    help="Number of documents to sample for each column value (e.g., 100).",
)
@click.option(
    "--seed",
    default=42,
    type=int,
    show_default=True,
    help="Seed value for random operations to ensure reproducibility.",
)
def sample_from_hf_dataset_cli(
    dataset_name: str,
    dataset_split: str,
    output_file_path: Path,
    column_name: str,
    column_values: str,
    num_docs_per_value: int,
    seed: int,
):
    column_values_list = [x.strip() for x in column_values.split(",")]
    sample_from_hf_dataset(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        output_file_path=output_file_path,
        column_name=column_name,
        column_values=column_values_list,
        num_docs_per_value=num_docs_per_value,
        seed=seed,
    )


@main.command(name="upload_file_to_hf")
@click.option(
    "--file_path",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    required=True,
    help="The local path to the file to be uploaded.",
)
@click.option(
    "--hf_repo_path",
    type=str,
    required=True,
    help="The path in the Hugging Face repository where the file will be stored.",
)
@click.option(
    "--hf_repo_id",
    type=str,
    required=True,
    help="The ID of the Hugging Face repository.",
)
@click.option(
    "--repo_type",
    type=str,
    default="dataset",
    show_default=True,
    help="The type of the repository (default is 'dataset').",
)
@click.option(
    "--hf_token",
    type=str,
    default=os.environ.get("HF_TOKEN", ""),
    show_default=True,
    help="The Hugging Face authentication token (default is taken from the environment variable 'HF_TOKEN').",
)
def upload_file_to_hf_cli(file_path: Path, hf_repo_path: str, hf_repo_id: str, repo_type: str, hf_token: str):
    """Upload a file to the Hugging Face Hub."""
    upload_file_to_hf(
        file_path=str(file_path),
        hf_repo_path=hf_repo_path,
        hf_repo_id=hf_repo_id,
        repo_type=repo_type,
        hf_token=hf_token,
    )


@main.command(name="merge_and_sort_jsonl_files_cli")
@click.option(
    "--input_folder_path",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to the input folder containing JSONL files.",
)
@click.option("--file-name-delimiter", type=str, required=True, help="Delimiter used to split the file names.")
@click.option(
    "--file-name-keep-idx",
    type=str,
    required=True,
    help="Comma-separated list of indices to keep from the split file names.",
)
@click.option("--document-key", type=str, required=True, help="The key used to sort documents in the JSONL files.")
def merge_and_sort_jsonl_files_cli(
    input_folder_path: Path, file_name_delimiter: str, file_name_keep_idx: str, document_key: str
):
    """Merge and sort JSONL files in a directory by a specific key."""
    # Parse file_name_keep_idx into a list of integers
    file_name_keep_idx_list = [int(idx.strip()) for idx in file_name_keep_idx.split(",")]

    # Call the main function to process JSONL files
    merge_and_sort_jsonl_files(
        directory=input_folder_path,
        file_name_delimiter=file_name_delimiter,
        file_name_keep_idx=file_name_keep_idx_list,
        document_key=document_key,
    )


@main.command(name="count_words_in_jsonl_files_cli")
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path),
    required=True,
    help="Path to the directory containing JSONL files.",
)
@click.option(
    "--output-file",
    type=click.Path(file_okay=True, dir_okay=False, writable=True, path_type=Path),
    required=True,
    help="Path to the output file (JSONL or YAML format) to save results.",
)
def count_words_in_jsonl_files_cli(directory: Path, output_file: Path) -> None:
    """
    CLI wrapper to count words in all JSONL files within a directory recursively and save the result.

    Args:
        directory (Path): Path to the directory to search for JSONL files.
        output_file (Path): Path to the output file (JSONL or YAML format) to save results.
    """
    run_word_count_jsonl_files(directory, output_file)


@main.command(name="deduplicate_jsonl")
@click.option(
    "--input_file",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the input JSONL file.",
)
@click.option(
    "--output_file",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to the output file with deduplicated entries.",
)
def deduplicate_jsonl_cli(input_file: Path, output_file: Path):
    """
    CLI command to deduplicate entries in a JSONL file based on 'doc_id' and 'text' fields.
    """
    deduplicate_jsonl(input_file_path=input_file, output_file_path=output_file)
    print(f"Processed {input_file} -> {output_file}")
    
    
@main.command(name="deduplicate_dir")
@click.option(
    "--input_dir",
    type=click_pathlib.Path(exists=True),
    required=True,
    help="Path to the directory with JSONL files.",
)
@click.option(
    "--output_dir",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to the output directory with deduplicated entries.",
)
def deduplicate_dir_cli(input_dir: Path, output_dir: Path):
    """
    CLI command to deduplicate entries in all JSONL files in a directory based on 'doc_id' and 'text' fields.
    """
    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Iterate over all JSONL files in the input directory
    for input_file in input_dir.glob("*.jsonl"):
        output_file = output_dir / input_file.name  # Keep the same filename in the output directory
        deduplicate_jsonl(input_file_path=input_file, output_file_path=output_file)
        print(f"Processed {input_file} -> {output_file}")
        

@main.command(name="convert_hf_dataset_to_jsonl")
@click.option(
    "--output_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to output file.",
)
@click.option(
    "--hf_dataset_name",
    type=str,
    help="Name of the Hugging Face dataset to download and convert.",
)
@click.option(
    "--hf_dataset_split",
    type=str,
    default="train",
    show_default=True,
    help="The split of the Hugging Face dataset that is used for conversion.",
)
def convert_hf_dataset_to_jsonl_cli(
    output_dir_path: Path,
    hf_dataset_name: str,
    hf_dataset_split: str,
):
    """Convert the FineWeb dataset into JSONL format and create train/val/test splits.

    This command downloads the dataset from Hugging Face, converts it to JSONL format,
    creates multiple versions with different score transformations, and splits the data
    into train/validation/test sets.
    """
    # download data and create single score file
    convert_hf_dataset_to_jsonl(
        hf_dataset_name=hf_dataset_name,
        output_dir_path=output_dir_path,
        hf_dataset_split=hf_dataset_split,
    )


@main.command(name="create_train_val_test_split")
@click.option(
    "--input_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to input file.",
)
@click.option(
    "--output_dir_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to output directory.",
)
@click.option(
    "--split_ratio",
    type=str,
    help="Comma seprated train, validation, test split raio.",
)
def create_train_val_test_split_cli(
    input_file_path: Path,
    output_dir_path: Path,
    split_ratio: str,
):
    train_ratio, val_ratio, test_ratio = (float(ratio) for ratio in split_ratio.split(","))
    split_dataset(
        input_file_path=input_file_path,
        output_dir_path=output_dir_path,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


@click.command(name="apply_score_transforms")
@click.option(
    "--input_file_path",
    type=click_pathlib.Path(exists=False, resolve_path=True),
    required=True,
    help="Path to input file.",
)
@click.option(
    "--output_file_path",
    type=click_pathlib.Path(exists=False, resolve_path=True),
    required=True,
    help="Path to input file.",
)
def apply_score_transforms_cli(input_file_path: Path, output_file_path: Path) -> None:
    """CLI command to apply score transformations and save results."""

    def get_transform_functions():
        """Returns a list of transformation functions for scores."""
        return [
            # TODO: Assign names to the transformations
            ("shift_up_capped", lambda x: min(x + 1, 5)),  # Shift up by 1, cap at 5
            ("add_noise_clamped", lambda x: min(max(x + random.uniform(-0.5, 0.5), 0), 5)),  # Add noise, clamp to [0,5]
            ("binary_threshold", lambda x: 1 if x >= 3 else 0),  # Binary threshold at 3
        ]

    # Apply transformations
    apply_score_transforms(
        input_file_path=input_file_path,
        output_path=output_file_path,
        transform_fns=get_transform_functions(),
    )


def _get_translator_helper(translation_service: str, ignore_tag_text: Optional[str] = None):
    translation_service_type = TranslationServiceType[translation_service]
    return TranslatorFactory.get_translator(
        translation_service_type=translation_service_type, ignore_tag_text=ignore_tag_text
    )


def _get_target_language_codes_list_helper(target_language_codes: str) -> list[str]:
    return [lang_code.strip().lower() for lang_code in target_language_codes.split(",")]


# @main.command(name="inference_pipeline")
# @click.option(
#     "--config_file_path",
#     type=click_pathlib.Path(exists=True),
#     required=True,
#     help="Path to a file with the YAML config file.",
# )
# def entry_inference_pipeline(config_file_path: Path):
#     run_pipeline(config_file_path)


if __name__ == "__main__":
    main()
