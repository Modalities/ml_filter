import os
import sys
from pathlib import Path

import click
import click_pathlib

from ml_filter.llm_client import LLMClient

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
    llm_service = LLMClient(config_file_path=config_file_path)
    llm_service.run()


if __name__ == "__main__":
    main()
