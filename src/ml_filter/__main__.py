import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import click_pathlib

from ml_filter.analysis.interrater_reliability import compute_interrater_reliability_metrics
from ml_filter.analysis.plot_score_distributions import plot_differences_in_scores, plot_scores
from ml_filter.classifier_training_pipeline import ClassifierTrainingPipeline
from ml_filter.compare_experiments import compare_experiments
from ml_filter.llm_client import LLMClient
from ml_filter.sample_from_hf_dataset import sample_from_hf_dataset
from ml_filter.translate import TranslationServiceType, TranslatorFactory
from ml_filter.utils.chunk_data import chunk_jsonl
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
    required=False,
    help="""
        Specifies how scores for a document from the same file are aggregated. 
        If not set, no aggregation will be done (used for individual annotator analysis).
        Supported values:
        - "mean": Compute the average score.
        - "max": Use the maximum score.
        - "min": Use the minimum score.
        - "majority": Use the score that was voted the most. If there is a tie, take the average of the winners.
    """,
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
    if experiment_id is None:
        with open(config_file_path, "rb") as f:
            hash_value = hashlib.sha256(f.read()).hexdigest()[:8]
        experiment_id = datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + f"__{hash_value}"
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


@main.command(name="train_classifier")
@click.option(
    "--config_file_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to the training config file.",
)
def entry_train_classifier(config_file_path: Path):
    classifier_pipeline = ClassifierTrainingPipeline(config_file_path=config_file_path)
    classifier_pipeline.train_classifier()


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
def plot_scores_cli(path_to_files: tuple[Path], output_dir: str, aggregation: Optional[str] = None) -> None:
    """Plot the differences in scores."""
    path_to_files = [Path(p) for p in path_to_files]
    plot_scores(path_to_files=path_to_files, output_dir=Path(output_dir), aggregation=aggregation)
    plot_differences_in_scores(path_to_files=path_to_files, output_dir=Path(output_dir), aggregation=aggregation)


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
    help="Name of the Hugging Face dataset to sample from (e.g., 'HuggingFaceFW/fineweb-edu-llama3-annotations')."
)
@click.option(
    "--dataset_split",
    required=True,
    type=str,
    help="The split of the Hugging Face dataset that is used for sampling (e.g., 'train')."
)
@click.option(
    "--output_file_path",
    required=True,
    type=click.Path(),
    help="Path to save the sampled data as a JSON file (e.g., 'output.json')."
)
@click.option(
    "--hf_repo_path",
    required=True,
    type=str,
    help="Path in the Hugging Face Hub repository where the file will be stored (e.g., 'dataset/output.json')."
)
@click.option(
    "--hf_repo_id",
    required=True,
    type=str,
    help="Hugging Face repository ID (e.g., 'username/repository')."
)
@click.option(
    "--column_name",
    required=True,
    type=str,
    help="Column in the dataset used for filtering (e.g., 'score')."
)
@click.option(
    "--relevant_classes",
    required=True,
    type=str,
    help="Comma-separated list of relevant class values to sample.",
)
@click.option(
    "--num_docs_per_class",
    required=True,
    type=int,
    help="Number of documents to sample for each class (e.g., 100)."
)
@click.option(
    "--seed",
    default=42,
    type=int,
    show_default=True,
    help="Seed value for random operations to ensure reproducibility."
)
def sample_from_hf_dataset_cli(
    dataset_name: str,
    dataset_split: str,
    output_file_path: str,
    hf_repo_id: str,
    hf_repo_path: str,
    column_name: str,
    relevant_classes: tuple[int],
    num_docs_per_class: int,
    seed: int
):
    relevant_classes_list = [int(x.strip()) for x in relevant_classes.split(",")]
    sample_from_hf_dataset(
        dataset_name=dataset_name,
        dataset_split=dataset_split,
        output_file_path=output_file_path,
        hf_repo_path=hf_repo_path,
        hf_repo_id=hf_repo_id,
        column_name=column_name,
        relevant_classes=relevant_classes_list,
        num_docs_per_class=num_docs_per_class,
        seed=seed,
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


def _get_translator_helper(translation_service: str, ignore_tag_text: Optional[str] = None):
    translation_service_type = TranslationServiceType[translation_service]
    return TranslatorFactory.get_translator(
        translation_service_type=translation_service_type, ignore_tag_text=ignore_tag_text
    )


def _get_target_language_codes_list_helper(target_language_codes: str) -> list[str]:
    return [lang_code.strip().lower() for lang_code in target_language_codes.split(",")]


if __name__ == "__main__":
    main()
