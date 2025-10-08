from pathlib import Path
from typing import Optional

from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from omegaconf import OmegaConf
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
    tasks: int = 1
    local_tasks: int = 1
    local_rank_offset: int = 0
    workers: int = -1
    logging_dir: Optional[str] = None


class SlurmExecutionSettings(BaseModel):
    tasks: int = 1
    time: str = "00:30:00"
    partition: str = "default"
    cpus_per_task: int = 4
    mem_per_cpu_gb: int = 8
    workers: int = -1
    job_name: str = "annotation_pipeline"
    qos: str = "normal"
    env_command: Optional[str] = None
    condaenv: Optional[str] = None
    venv_path: Optional[str] = None
    sbatch_args: Optional[dict[str, str | int | float | bool]] = None
    max_array_size: int = 1001
    depends_job_id: Optional[str] = None
    job_id_position: int = -1
    logging_dir: Optional[str] = None
    skip_completed: bool = True
    slurm_logs_folder: Optional[str] = None
    mail_type: str = "ALL"
    mail_user: Optional[str] = None
    requeue: bool = True
    srun_args: Optional[dict[str, str | int | float | bool]] = None
    tasks_per_job: int = 1

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
    output_dir: Path = Field(..., description="Output directory for annotated JSONL files.")
    regression_head_checkpoints: dict[str, str] = Field(..., description="Mapping of model names to head checkpoint paths.")
    batch_size: int = Field(512, description="Batch size for processing embeddings.")
    
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
        rs = raw.get("running_on_slurm", False) if running_on_slurm is None else running_on_slurm
        slurm_settings = raw.get("slurm_settings", None)
        local_section = raw.get("local_settings", None)

        params = AnnotationPipelineParameters(
            embeddings_directory=params_cfg["embeddings_directory"],
            output_dir=params_cfg["output_dir"],
            regression_head_checkpoints=params_cfg["regression_head_checkpoints"],
            batch_size=params_cfg.get("batch_size", 512),
        )

        local_settings_obj = None
        if not rs and isinstance(local_section, dict):
            local_settings_obj = LocalExecutionSettings(**local_section)

        builder_kwargs = {"params": params, "running_on_slurm": rs}
        if local_settings_obj:
            builder_kwargs["local_settings"] = local_settings_obj
        if rs and slurm_settings:
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
                    adapter=stats_adapter,
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
