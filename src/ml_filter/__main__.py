from datetime import datetime
import hashlib
from pathlib import Path
from typing import Optional

import click
import click_pathlib

from ml_filter.classifier_training_pipeline import ClassifierTrainingPipeline
from ml_filter.llm_client import LLMClient
from ml_filter.utils.chunk_data import chunk_jsonl

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
def entry_point_score_documents(config_file_path: Path, experiment_id: Optional[str] = None):
    if experiment_id is None:
        with open(config_file_path, "rb") as f:
            hash_value = hashlib.file_digest(f, "sha256").hexdigest()[:8]
        experiment_id = datetime.now().strftime("%Y-%m-%d__%H-%M-%S") + f"__{hash_value}"

    llm_service = LLMClient(config_file_path=config_file_path, experiment_id=experiment_id)
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

if __name__ == "__main__":
    main()
