"""Config-driven quantile pipeline for per-language JSONL filtering."""

from pathlib import Path

from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from omegaconf import DictConfig as _DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml_filter.ablations.quantile_steps import QuantileJsonlReader, raw_data_adapter


class QuantilePipelineParameters(BaseModel):
    """Configuration parameters for the quantile pipeline."""

    input_dir: str = Field(..., description="Directory containing JSONL files.")
    glob_pattern: str = Field(..., description="Glob for selecting JSONL files.")
    output_dir: Path = Field(..., description="Root output directory.")
    compression: str | None = Field(..., description="Compression for input JSONL files (infer/gzip/zstd/None).")
    output_compression: str | None = Field(
        None,
        description="Compression for output JSONL files (infer/gzip/None).",
    )
    score_field: str = Field(..., description="Field name that stores scores.")
    selection_quantile: float = Field(..., description="Top fraction of data to keep (e.g., 0.2 keeps top 20%).")
    report_filename: str = Field("quantile_report.yaml", description="Filename for the YAML report.")
    quantile_data_dir: str = Field("quantile_data", description="Subdirectory for filtered outputs.")

    @property
    def report_path(self) -> Path:
        return self.output_dir / self.report_filename

    @property
    def quantile_output_dir(self) -> Path:
        return self.output_dir / self.quantile_data_dir


class LocalExecutionSettings(BaseModel):
    """Local executor settings for datatrove."""

    tasks: int = 1
    local_tasks: int = 1
    local_rank_offset: int = 0
    workers: int = -1
    logging_dir: str | None = None


class SlurmExecutionSettings(BaseModel):
    """Slurm executor settings for datatrove."""

    tasks: int = 1
    time: str = "00:30:00"
    partition: str = "default"
    cpus_per_task: int = 4
    mem_per_cpu_gb: int = 8
    workers: int = -1
    job_name: str = "quantile_pipeline"
    qos: str = "normal"
    env_command: str | None = None
    condaenv: str | None = None
    venv_path: str | None = None
    sbatch_args: dict[str, str | int | float | bool] | None = None
    max_array_size: int = 1001
    depends_job_id: str | None = None
    job_id_position: int = -1
    logging_dir: str | None = None
    skip_completed: bool = True
    slurm_logs_folder: str | None = None
    mail_type: str = "ALL"
    mail_user: str | None = None
    requeue: bool = True
    srun_args: dict[str, str | int | float | bool] | None = None
    tasks_per_job: int = 1

    @model_validator(mode="before")
    def _normalize_sbatch(cls, values):  # type: ignore[override]
        sbatch_args = values.get("sbatch_args") or {}
        if isinstance(sbatch_args, _DictConfig):
            sbatch_args = OmegaConf.to_container(sbatch_args, resolve=True)
        if not isinstance(sbatch_args, dict):
            raise TypeError(f"sbatch_args must be a mapping if provided (got type {type(sbatch_args)})")
        values["sbatch_args"] = sbatch_args
        return values


class QuantilePipelineBuilder(BaseSettings):
    """Builds the quantile pipeline and executor from a YAML config."""

    model_config = SettingsConfigDict(env_prefix="quantile_pipeline_", env_nested_delimiter="__")

    params: QuantilePipelineParameters
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
    ) -> "QuantilePipelineBuilder":
        """Load config and build a pipeline builder instance."""
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = OmegaConf.load(path)

        if "params" not in raw:
            raise ValueError("YAML must contain a top-level 'params:' section.")

        params_cfg = raw["params"]
        if isinstance(params_cfg, _DictConfig):
            params_cfg = OmegaConf.to_container(params_cfg, resolve=True)
        if not isinstance(params_cfg, dict):
            raise TypeError("`params` section must be a mapping.")

        rs = raw.get("running_on_slurm", False) if running_on_slurm is None else running_on_slurm
        slurm_settings = raw.get("slurm_settings", None)
        local_section = raw.get("local_settings", None)

        if isinstance(local_section, _DictConfig):
            local_section = OmegaConf.to_container(local_section, resolve=True)
        if isinstance(slurm_settings, _DictConfig):
            slurm_settings = OmegaConf.to_container(slurm_settings, resolve=True)

        def _p(name: str, default=None):
            return params_cfg.get(name, default)

        params = QuantilePipelineParameters(
            input_dir=_p("input_dir"),
            glob_pattern=_p("glob_pattern"),
            output_dir=_p("output_dir"),
            compression=_p("compression"),
            output_compression=_p("output_compression", None),
            score_field=_p("score_field", "scores"),
            selection_quantile=_p("selection_quantile"),
            report_filename=_p("report_filename", "quantile_report.yaml"),
            quantile_data_dir=_p("quantile_data_dir", "quantile_data"),
        )

        builder_kwargs = {"params": params, "running_on_slurm": rs}
        if rs:
            if slurm_settings is not None:
                builder_kwargs["slurm_settings"] = SlurmExecutionSettings(**slurm_settings)
        else:
            if isinstance(local_section, dict):
                builder_kwargs["local_settings"] = LocalExecutionSettings(
                    **{k: v for k, v in local_section.items() if k in LocalExecutionSettings.model_fields}
                )

        return cls(**builder_kwargs)

    def build_pipeline(self) -> list[PipelineStep]:
        """Construct the datatrove pipeline steps."""
        p = self.params
        return [
            QuantileJsonlReader(
                data_folder=p.input_dir,
                glob_pattern=p.glob_pattern,
                compression=p.compression,
                score_field=p.score_field,
                selection_quantile=p.selection_quantile,
                report_path=p.report_path,
            ),
            JsonlWriter(
                output_folder=str(p.quantile_output_dir),
                output_filename="${language}/${source_filename}.jsonl",
                adapter=raw_data_adapter,
                compression=p.output_compression,
            ),
        ]

    def build_executor(self) -> LocalPipelineExecutor | SlurmPipelineExecutor:
        """Create the local or Slurm executor based on config."""
        pipeline = self.build_pipeline()
        if self.running_on_slurm:
            print("Running Slurm Quantile Pipeline Executor")
            return SlurmPipelineExecutor(pipeline=pipeline, **self.slurm_settings.model_dump())
        print("Running Local Quantile Pipeline Executor")
        return LocalPipelineExecutor(pipeline=pipeline, **self.local_settings.model_dump())


def run_quantile_pipeline(config_file_path: Path):
    """Run quantile pipeline directly from YAML file."""
    builder = QuantilePipelineBuilder.from_yaml(config_file_path)
    executor = builder.build_executor()
    executor.run()
