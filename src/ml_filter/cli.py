import os
import sys
from pathlib import Path

import click
import click_pathlib
from classifier_training_pipeline import ClassifierTrainingPipeline
from llm_client import LLMClient
from translate import deepl_translate, write_output

sys.path.append(os.path.join(os.getcwd(), "src"))


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
def entry_point_score_documents(config_file_path: Path):
    llm_service = LLMClient(config_file_path=config_file_path)
    llm_service.run()


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


@main.command(name="deepl_translate_cli")
@click.option(
    "--input_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to the input file.",
)
@click.option(
    "--output_path",
    type=click_pathlib.Path(exists=False),
    required=True,
    help="Path to the output file.",
)
@click.option(
    "--api_key",
    type=str,
    required=True,
    help="Authentication key for DeepL.",
)
@click.option(
    "--source_language",
    type=str,
    required=True,
    help="Path to the output file.",
)
@click.argument("languages", nargs=-1)
def deepl_translate_cli(input_path: Path, output_path: Path, api_key: str, source_language: str, languages: list[str]):
    translated_data = deepl_translate(
        input_path=input_path, api_key=api_key, source_language=source_language, languages=languages
    )
    write_output(output_path=output_path, data=translated_data)


if __name__ == "__main__":
    main()
