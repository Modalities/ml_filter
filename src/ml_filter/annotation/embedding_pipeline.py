from pathlib import Path

from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from omegaconf import OmegaConf

from ml_filter.annotation.datatrove_jql_annotator import HDF5Writer, JQLEmbedder, JQLJsonlReader

import os
import sys
from pathlib import Path
from typing import Optional

from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict, YamlConfigSettingsSource
from ml_filter.annotation.datatrove_jql_annotator import HDF5Writer, JQLEmbedder, JQLJsonlReader


class EmbeddingPipelineParameters(BaseModel):
    input_dir: str = Field(..., description="Directory containing JSONL files.")
    csv_hashmap_path: Path = Field(..., description="CSV mapping file paths to md5 hashes.")
    glob_pattern: str = Field(..., description="Glob for selecting JSONL files.")
    output_dir: Path = Field(..., description="Root output directory.")
    embedding_dir: str = Field(..., description="Subdirectory for embedding outputs.")
    embedding_model: str = Field(..., description="Embedding model identifier.")
    hdf5_dataset_name: str = Field(..., description="Dataset/group name in HDF5 output.")
    batch_size: int = Field(..., description="Embedding batch size.")
    writer_batch_size: int = Field(..., description="Batch size for flush to disk in writer.")
    max_length: int = Field(..., description="Max token length.")
    padding: bool | str = Field(..., description="Padding strategy.")
    truncation: bool | str = Field(..., description="Truncation strategy.")
    save_labels: bool = Field(..., description="Copy score->label if present when writing.")

    @property
    def embedding_output_dir(self) -> Path:
        return self.output_dir / self.embedding_dir


class LocalExecutionSettings(BaseModel):
    tasks: int = 1
    local_tasks: int = 1
    local_rank_offset: int = 0
    workers: int = -1
    logging_dir: str | None = None


class SlurmExecutionSettings(BaseModel):
    tasks: int = 1
    time: str = "00:30:00"
    partition: str = "default"
    cpus_per_task: int = 4
    mem_per_cpu_gb: int = 8
    workers: int = -1
    job_name: str = "embedding_pipeline"
    qos: str = "normal"
    env_command: str | None = None
    condaenv: str | None = None
    venv_path: str | None = None
    # Allow users to supply any sbatch arg (e.g. nodes, ntasks, gres, account, output, error, gpus-per-task, etc.)
    # using either snake_case or dash-case. Primitive values get coerced to strings.
    sbatch_args: dict[str, str | int | float | bool] | None = None
    max_array_size: int = 1001
    depends_job_id: str | None = None
    job_id_position: int = -1
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
    srun_args: dict[str, str | int | float | bool] | None = None
    tasks_per_job: int = 1

    @model_validator(mode="before")
    def _normalize_sbatch(cls, values):  # type: ignore[override]
        """Normalize sbatch_args only.

        - Accept numeric/bool types and coerce to string
        - Fold common top-level keys (output, error, gpus_per_task) into sbatch_args
        - Convert snake_case keys to dash-case
        """
        from omegaconf import DictConfig as _DictConfig  # local import

        sbatch_args = values.get("sbatch_args") or {}
        if isinstance(sbatch_args, _DictConfig):
            sbatch_args = OmegaConf.to_container(sbatch_args, resolve=True)  # type: ignore[arg-type]
        if not isinstance(sbatch_args, dict):
            raise TypeError(f"sbatch_args must be a mapping if provided (got type {type(sbatch_args)})")

        values["sbatch_args"] = sbatch_args
        return values


class EmbeddingPipelineBuilder(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="embedding_pipeline_", env_nested_delimiter="__")

    params: EmbeddingPipelineParameters
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
            self.local_settings.logging_dir = str(self.params.output_dir / "logs")
        if self.slurm_settings is not None and self.slurm_settings.logging_dir is None:
            self.slurm_settings.logging_dir = str(self.params.output_dir / "logs")
        return self

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        running_on_slurm: bool | None = None,
    ) -> "EmbeddingPipelineBuilder":
        """Create a builder directly from a YAML file.

        Supports two schema styles:
          1. Legacy flat (keys like input_dir, embedding_model, tasks, ...)
          2. Builder style with a top-level `params:` mapping (and optional running_on_slurm/slurm_settings)

        Args:
            path: Path to YAML file
            running_on_slurm: Optional override execution mode. If None, value will be read from YAML
                (key: `running_on_slurm`; defaults to False when absent). If a boolean is provided it
                takes precedence over the YAML value.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = OmegaConf.load(path)

        # Detect builder vs legacy style
        if "params" in raw:  # builder style
            params_cfg = raw["params"]  # still a DictConfig
            OmegaConf.resolve(params_cfg)  # resolves all ${...} references
            # Respect explicit override only when the caller supplies a value; otherwise honor YAML
            rs = raw.get("running_on_slurm", False) if running_on_slurm is None else running_on_slurm
            slurm_settings = raw.get("slurm_settings", None)
            local_section = raw.get("local_settings", None)
        else:  # legacy flat style (your current YAML)
            raise DeprecationWarning(
                "Legacy flat config style is deprecated. Please migrate to builder style with a top-level 'params:' section."
            )

        # Simple interpolation for ${dataset_name} tokens in legacy-style values
        dataset_name = params_cfg.get("dataset_name") if isinstance(params_cfg, dict) else None
        if dataset_name:
            for k, v in list(params_cfg.items()):  # type: ignore[attr-defined]
                if isinstance(v, str) and "${dataset_name}" in v:
                    params_cfg[k] = v.replace("${dataset_name}", str(dataset_name))

        def _p(name: str, default=None):
            return params_cfg.get(name, default)

        params = EmbeddingPipelineParameters(
            input_dir=_p("input_dir"),
            csv_hashmap_path=_p("csv_hashmap_path"),
            glob_pattern=_p("glob_pattern", "*.jsonl"),
            output_dir=_p("output_dir"),
            embedding_dir=_p("embedding_dir", "embeddings"),
            embedding_model=_p("embedding_model", "Snowflake/snowflake-arctic-embed-m-v2.0"),
            hdf5_dataset_name=_p("hdf5_dataset_name", "train"),
            batch_size=_p("batch_size", 256),
            writer_batch_size=_p("writer_batch_size", 1000),
            max_length=_p("max_length", 8192),
            padding=_p("padding", True),
            truncation=_p("truncation", True),
            save_labels=_p("save_labels", True),
            tasks=_p("tasks", local_section.get("tasks") if isinstance(local_section, dict) else 1) or 1,
            workers=_p("workers", local_section.get("workers") if isinstance(local_section, dict) else -1) or -1,
            local_tasks=_p("local_tasks", local_section.get("local_tasks") if isinstance(local_section, dict) else 1) or 1,
            local_rank_offset=_p(
                "local_rank_offset",
                local_section.get("local_rank_offset") if isinstance(local_section, dict) else 0,
            ) or 0,
        )

        # Build explicit local settings if provided
        local_settings_obj = None
        if not rs and isinstance(local_section, dict):
            # None fallback logic ensures defaults if missing
            local_settings_obj = LocalExecutionSettings(
                tasks=local_section.get("tasks", params.tasks),
                local_tasks=local_section.get("local_tasks", params.local_tasks),
                local_rank_offset=local_section.get("local_rank_offset", params.local_rank_offset),
                workers=local_section.get("workers", params.workers),
            )

        builder_kwargs = {"params": params, "running_on_slurm": rs}
        if local_settings_obj is not None:
            builder_kwargs["local_settings"] = local_settings_obj
        if rs and slurm_settings:
            # Only include recognized slurm settings keys
            builder_kwargs["slurm_settings"] = SlurmExecutionSettings(**slurm_settings)
        return cls(**builder_kwargs)  # type: ignore[arg-type]

    def build_pipeline(self) -> list[PipelineStep]:
        p = self.params
        pipeline: list[PipelineStep] = [
            JQLJsonlReader(
                data_folder=p.input_dir,
                csv_hashmap=p.csv_hashmap_path,
                glob_pattern=p.glob_pattern,
                save_labels=p.save_labels,
            ),
            JQLEmbedder(
                embedder_model_id=p.embedding_model,
                batch_size=p.batch_size,
                max_length=p.max_length,
                padding=p.padding,
                truncation=p.truncation,
                stats_writer=HDF5Writer(
                    output_folder=str(p.embedding_output_dir),
                    output_filename="${source_filename}.h5",
                    dataset_name=p.hdf5_dataset_name,
                    batch_size=p.writer_batch_size,
                    save_labels=p.save_labels,
                ),
            ),
        ]
        return pipeline

    def build_executor(self) -> LocalPipelineExecutor | SlurmPipelineExecutor:
        pipeline = self.build_pipeline()
        if self.running_on_slurm:
            print("Running Slurm Pipeline Executor")
            return SlurmPipelineExecutor(pipeline=pipeline, **self.slurm_settings.model_dump())
        print("Running Local Pipeline Executor")
        return LocalPipelineExecutor(pipeline=pipeline, **self.local_settings.model_dump())



def run_embedding_pipeline(config_file_path: Path):
    """Run embedding pipeline directly from YAML file."""
    builder = EmbeddingPipelineBuilder.from_yaml(config_file_path)
    executor = builder.build_executor()
    executor.run()