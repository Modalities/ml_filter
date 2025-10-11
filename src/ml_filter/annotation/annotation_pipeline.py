from functools import partial
from pathlib import Path
from typing import Optional

from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from omegaconf import OmegaConf
from omegaconf import DictConfig as _DictConfig
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml_filter.annotation.datatrove_jql_annotator import (
    JQLHead,
    stats_adapter,
    JQLEmbeddingReader,
)

# ---------------------------------------------------------------------------
# Base Configuration Classes (can be reused from embedding pipeline)
# ---------------------------------------------------------------------------

class LocalExecutionSettings(BaseModel):
    tasks: int
    local_tasks: int
    local_rank_offset: int
    workers: int
    logging_dir: Optional[str]


class SlurmExecutionSettings(BaseModel):
    tasks: int
    time: str
    partition: str
    cpus_per_task: int
    mem_per_cpu_gb: int
    workers: int
    job_name: str
    qos: str
    env_command: Optional[str]
    condaenv: Optional[str]
    venv_path: Optional[str]
    sbatch_args: Optional[dict[str, str | int | float | bool]]
    max_array_size: int
    depends_job_id: Optional[str]
    job_id_position: int
    logging_dir: Optional[str]
    skip_completed: bool
    slurm_logs_folder: Optional[str]
    mail_type: str
    mail_user: Optional[str]
    requeue: bool
    srun_args: Optional[dict[str, str | int | float | bool]]
    tasks_per_job: int

    @model_validator(mode="before")
    def _normalize_sbatch(cls, values):
        """Normalize sbatch_args only.

        - Accept numeric/bool types and coerce to string
        - Fold common top-level keys (output, error, gpus_per_task) into sbatch_args
        - Convert snake_case keys to dash-case
        """
        from omegaconf import DictConfig as _DictConfig

        sbatch_args = values.get("sbatch_args") or {}
        if isinstance(sbatch_args, _DictConfig):
            sbatch_args = OmegaConf.to_container(sbatch_args, resolve=True)
        if not isinstance(sbatch_args, dict):
            raise TypeError(f"sbatch_args must be a mapping if provided (got type {type(sbatch_args)})")

        values["sbatch_args"] = sbatch_args
        return values

# ---------------------------------------------------------------------------
# Annotation Pipeline Configuration
# ---------------------------------------------------------------------------

class AnnotationPipelineParameters(BaseModel):
    embeddings_directory: str = Field(..., description="Path to directory containing HDF5 embedding files.")
    output_keys: list[str] = Field(..., description="List of metadata keys to include in the annotated output files.")
    output_dir: Path = Field(..., description="Output directory for annotated JSONL files.")
    regression_head_checkpoints: dict[str, str] = Field(..., description="Mapping of model names to head checkpoint paths.")
    batch_size: int = Field(..., description="Batch size for processing embeddings.")
    
    @property
    def annotated_output_dir(self) -> Path:
        return self.output_dir / "annotated_data"


# ---------------------------------------------------------------------------
# Annotation Pipeline Builder
# ---------------------------------------------------------------------------

class AnnotationPipelineBuilder(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="annotation_pipeline_", env_nested_delimiter="__")

    params: AnnotationPipelineParameters
    running_on_slurm: bool = False
    local_settings: Optional[LocalExecutionSettings] = None
    slurm_settings: Optional[SlurmExecutionSettings] = None

    # --- Validators ---
    @model_validator(mode="after")
    def validate_execution_mode(self):
        if self.running_on_slurm:
            if self.local_settings is not None:
                raise ValueError("Running on Slurm requires only slurm_settings, not local_settings.")
            if self.slurm_settings is None:
                raise ValueError("running_on_slurm=True requires 'slurm_settings' section.")
        else:
            if self.slurm_settings is not None:
                raise ValueError("Running locally requires only local_settings, not slurm_settings.")
            if self.local_settings is None:
                raise ValueError("running_on_slurm=False requires 'local_settings' section.")
        return self

    @model_validator(mode="after")
    def set_logging_dir(self):
        if self.local_settings and self.local_settings.logging_dir is None:
            self.local_settings.logging_dir = str(self.params.output_dir / "logs")
        if self.slurm_settings and self.slurm_settings.logging_dir is None:
            self.slurm_settings.logging_dir = str(self.params.output_dir / "logs")
        return self

    # --- YAML Loader ---
    @classmethod
    def from_yaml(cls, path: Path, running_on_slurm: Optional[bool] = None) -> "AnnotationPipelineBuilder":
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = OmegaConf.load(path)

        if "params" not in raw:
            raise ValueError("YAML must contain a top-level 'params:' section (builder-style schema).")

        params_cfg = raw["params"]
        rs = raw.get("running_on_slurm") if running_on_slurm is None else running_on_slurm
        if rs is None:
            raise ValueError("YAML must specify 'running_on_slurm'.")
        slurm_settings = raw.get("slurm_settings", None)
        local_section = raw.get("local_settings", None)

        params = AnnotationPipelineParameters(
            embeddings_directory=params_cfg["embeddings_directory"],
            output_dir=params_cfg["output_dir"],
            output_keys=params_cfg["output_keys"],
            regression_head_checkpoints=params_cfg["regression_head_checkpoints"],
            batch_size=params_cfg["batch_size"],
        )

        local_settings_obj = None
        if not rs:
            if not isinstance(local_section, _DictConfig):
                raise ValueError("Local run requires 'local_settings' section in YAML.")
            local_settings_obj = LocalExecutionSettings(**local_section)
        else:
            if slurm_settings is None:
                raise ValueError("Slurm run requires 'slurm_settings' section in YAML.")

        builder_kwargs = {"params": params, "running_on_slurm": rs}
        if not rs:
            builder_kwargs["local_settings"] = local_settings_obj
        else:
            builder_kwargs["slurm_settings"] = SlurmExecutionSettings(**slurm_settings)

        return cls(**builder_kwargs)

    # --- Build Pipeline ---
    def build_pipeline(self) -> list[PipelineStep]:
        p = self.params
        pipeline = [
            JQLEmbeddingReader(data_folder=p.embeddings_directory),
            JQLHead(
                regression_head_checkpoints=p.regression_head_checkpoints,
                batch_size=p.batch_size,
                stats_writer=JsonlWriter(
                    output_folder=str(p.annotated_output_dir),
                    output_filename="${source_filename}.jsonl",
                    adapter=partial(stats_adapter, output_keys=p.output_keys),
                    expand_metadata=True,
                ),
            ),
        ]
        return pipeline

    # --- Build Executor ---
    def build_executor(self) -> LocalPipelineExecutor | SlurmPipelineExecutor:
        pipeline = self.build_pipeline()
        if self.running_on_slurm:
            print("Running Slurm Annotation Pipeline Executor")
            return SlurmPipelineExecutor(pipeline=pipeline, **self.slurm_settings.model_dump())
        print("Running Local Annotation Pipeline Executor")
        return LocalPipelineExecutor(pipeline=pipeline, **self.local_settings.model_dump())


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def run_annotation_pipeline(config_file_path: Path):
    """Run the annotation pipeline directly from a YAML file."""
    builder = AnnotationPipelineBuilder.from_yaml(config_file_path)
    executor = builder.build_executor()
    executor.run()
