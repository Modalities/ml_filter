from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource

from ml_filter.data_processing.score_based_filtering.step_data_filtering import DataFiltering
from ml_filter.data_processing.score_based_filtering.step_score_parsing import ScoresParser


class FilterPipelineBuilder(BaseSettings):
    """Configuration parameters and building for the score-based filtering pipeline.
    This class defines the settings for running a data filtering pipeline that processes datasets based on scores.
    It includes parameters for both local and Slurm execution environments.
    The pipeline consists of steps for parsing scores and filtering datasets based on those scores.

    Besides initializing this class directly, it can also be configured using a YAML file or environment variables.
    The YAML file can be specified using the `FILTER_PIPELINE_YAML_FILE` environment variable.
    If no YAML file is provided, the class will use default settings and environment variables.
    """

    model_config = SettingsConfigDict(env_prefix="filter_pipeline_", env_nested_delimiter="__")

    # Pipeline configuration parameters
    params: FilterPipelineParameters

    # Execution parameters
    running_on_slurm: bool = False
    local_settings: LocalExecutionSettings | None = None
    slurm_settings: SlurmExecutionSettings | None = None

    @model_validator(mode="after")
    def slurm_vs_local(self):
        if self.running_on_slurm and self.local_settings is not None:
            raise ValueError("Running on Slurm requires slurm execution settings, not local settings.")
        if self.running_on_slurm and self.slurm_settings is None:
            self.slurm_settings = SlurmExecutionSettings()
        elif not self.running_on_slurm and self.slurm_settings is not None:
            raise ValueError("Running locally requires local execution settings, not Slurm settings.")
        if not self.running_on_slurm and self.local_settings is None:
            self.local_settings = LocalExecutionSettings()
        return self

    @model_validator(mode="after")
    def set_logging_dir(self):
        if self.local_settings is not None and self.local_settings.logging_dir is None:
            self.local_settings.logging_dir = str(self.params.output_folder / "logs")
        if self.slurm_settings is not None and self.slurm_settings.logging_dir is None:
            self.slurm_settings.logging_dir = str(self.params.output_folder / "logs")
        return self

    def build_pipeline_executor(self) -> LocalPipelineExecutor | SlurmPipelineExecutor:
        """Builds the appropriate pipeline executor based on the execution settings."""
        pipeline = self._build_pipeline()
        if self.running_on_slurm:
            return SlurmPipelineExecutor(pipeline=pipeline, **self.slurm_settings.model_dump())
        else:
            return LocalPipelineExecutor(pipeline=pipeline, **self.local_settings.model_dump())

    def _build_pipeline(self) -> list[PipelineStep]:
        """Builds the pipeline based on the provided configuration."""
        return build_pipeline(
            score_path=self.params.score_path,
            tokenized_data_path=self.params.tokenized_data_path,
            output_folder=self.params.output_folder,
            thresholds=self.params.thresholds,
            hash_to_base_file_mapping_csv=self.params.hash_to_base_file_mapping_csv,
            base_file_prefix=self.params.base_file_prefix,
            tokenized_data_extension=self.params.tokenized_data_extension,
        )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            YamlConfigSettingsSource(settings_cls, yaml_file=os.getenv("FILTER_PIPELINE_YAML_FILE")),
            dotenv_settings,
            file_secret_settings,
        )


class FilterPipelineParameters(BaseModel):
    """Parameters for the score-based filtering pipeline."""

    score_path: Path = Field(..., description="The path to the directory containing JSONL files with scores.")
    tokenized_data_path: Path = Field(..., description="The path for the tokenized data files.")
    output_folder: Path = Field(..., description="The folder where the filtered datasets will be saved.")
    thresholds: dict[str, float] = Field(
        ..., description="Dictionary where keys are score names and values are thresholds to filter samples."
    )
    hash_to_base_file_mapping_csv: Path = Field(
        ..., description="CSV file mapping base file hashes to their corresponding paths."
    )
    base_file_prefix: Path = Field(
        default=Path(""),
        description="The prefix path for the raw/base files. This prefix will be removed "
        "when mapping from the raw files to the corresponding tokenized files",
    )
    tokenized_data_extension: str = Field(
        default=".pbin", description="The file extension for the tokenized data files."
    )


class LocalExecutionSettings(BaseModel):
    """Settings for running the pipeline locally."""

    tasks: int = 1
    local_tasks: int = 1
    local_rank_offset: int = 0
    logging_dir: str | None = None


class SlurmExecutionSettings(BaseModel):
    """Settings for running the pipeline on a Slurm cluster."""

    tasks: int = 1
    time: str = "00:15:00"
    partition: str = "default"
    account: str | None = None  # FIXME is this supported?
    cpus_per_task: int = 1
    mem_per_cpu_gb: int = 2
    workers: int = -1
    job_name: str = "data_processing"
    qos: str = "normal"
    env_command: str | None = None
    condaenv: str | None = None
    venv_path: str | None = None
    sbatch_args: dict[str, str] | None = None
    max_array_size: int = 1001
    depends_job_id: str | None = None
    job_id_position: int = -1
    # job_id_retriever: Callable | None = None
    logging_dir: str | None = None
    skip_completed: bool = True
    slurm_logs_folder: str | None = None
    max_array_launch_parallel: bool = False
    stagger_max_array_jobs: int = 0
    run_on_dependency_fail: bool = False
    randomize_start_duration: int = 0
    requeue_signals: tuple[str] | None = ("SIGUSR1",)
    mail_type: str = "ALL"
    mail_user: str | None = None
    requeue: bool = True
    srun_args: dict[str, str] | None = None
    tasks_per_job: int = 1


def run_pipeline(args: FilterPipelineBuilder) -> None:
    """Runs a datatrove pipeline to filter datasets based on scores.
    Args:
        args (PipelineArgs): The configuration parameters for the pipeline.
    """
    executor = args.build_pipeline_executor()
    executor.run()


def build_pipeline(
    score_path: Path,
    tokenized_data_path: Path,
    output_folder: Path,
    thresholds: dict[str, float],
    hash_to_base_file_mapping_csv: Path,
    base_file_prefix: Path = Path(""),
    tokenized_data_extension: str = ".pbin",
) -> list[PipelineStep]:
    """
    Builds a datatrove pipeline for filtering datasets based on scores.
    Args:
        score_path (Path): The path to the JSONL file containing scores.
        tokenized_data_path (Path): The path for the tokenized data files.
        output_folder (Path): The folder where the filtered datasets will be saved.
        thresholds (dict[str, float]): A dictionary where keys are score names and values are the
            thresholds to filter samples.
        hash_to_base_file_mapping_csv (Path): A CSV file mapping base file hashes to their corresponding paths.
        base_file_prefix (Path): The prefix path for the base files.
        tokenized_data_extension (str): The file extension for the tokenized data files.
    Returns:
        list[PipelineStep]: A list containing the pipeline steps for filtering datasets.
    """
    assert score_path.is_dir(), f"Score path {score_path} must be a directory."
    assert output_folder.is_dir(), f"Output folder {output_folder} must be a directory."
    assert len(thresholds) > 0, "At least one threshold must be provided."
    assert (
        hash_to_base_file_mapping_csv.is_file()
    ), f"Hash to base file mapping {hash_to_base_file_mapping_csv} must be a file."
    hash_to_base_file = read_hash_to_base_file_mapping(hash_to_base_file_mapping_csv)
    pipeline: list[PipelineStep] = [
        ScoresParser(
            data_folder=str(score_path),
            score_keys=list(thresholds.keys()),
            hash_to_base_file=hash_to_base_file,
            tokenized_data_path=tokenized_data_path,
            base_file_prefix=base_file_prefix,
            tokenized_data_extension=tokenized_data_extension,
        ),
        DataFiltering(
            output_folder=output_folder,
            thresholds=thresholds,
            tokenized_data_path=tokenized_data_path,
        ),
    ]
    return pipeline


def read_hash_to_base_file_mapping(csv_file: Path) -> dict[str, Path]:
    """
    Reads a CSV file containing a mapping from base file hashes to their corresponding paths.
    Args:
        csv_file (Path): The path to the CSV file.
    Returns:
        dict[str, Path]: A dictionary mapping base file hashes to their corresponding paths.
    """
    hash_to_base_file: dict[str, Path] = {}
    with open(csv_file, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            hash_to_base_file[row["md5"]] = Path(row["file_path"])
    return hash_to_base_file


if __name__ == "__main__":
    if len(sys.argv) > 1 or not (yaml_file := os.getenv("FILTER_PIPELINE_YAML_FILE")) or not os.path.isfile(yaml_file):
        print(
            "This script is intended to be used with a YAML configuration "
            "file set via the environment variable `FILTER_PIPELINE_YAML_FILE`.\n"
            "If you want to run it without a YAML file, please import from it "
            "and use the FilterPipelineBuilder class directly."
        )
        exit(1)
    args = FilterPipelineBuilder()
    run_pipeline(args)
