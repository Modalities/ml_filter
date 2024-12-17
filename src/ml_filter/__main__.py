import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import click_pathlib

from ml_filter.classifier_training_pipeline import ClassifierTrainingPipeline
from ml_filter.compare_experiments import compare_experiments
from ml_filter.llm_client import LLMClient
from ml_filter.translate import TranslationServiceType, TranslatorFactory
from ml_filter.utils.chunk_data import chunk_jsonl
from ml_filter.utils.manipulate_prompt import add_target_langauge_to_prompt
from ml_filter.utils.statistics import compute_num_words_and_chars_in_jsonl

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


@main.command(name="add_target_langauge_to_prompt_yaml")
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
def add_target_langauge_to_prompt_yaml(input_file_path: Path, output_dir: Path):
    add_target_langauge_to_prompt(input_file_path=input_file_path, output_dir=output_dir)


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


def _get_translator_helper(translation_service: str, ignore_tag_text: Optional[str] = None):
    translation_service_type = TranslationServiceType[translation_service]
    return TranslatorFactory.get_translator(
        translation_service_type=translation_service_type, ignore_tag_text=ignore_tag_text
    )


def _get_target_language_codes_list_helper(target_language_codes: str) -> list[str]:
    return [lang_code.strip().lower() for lang_code in target_language_codes.split(",")]


if __name__ == "__main__":
    main()
