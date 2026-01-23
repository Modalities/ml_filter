"""Config-driven quantile pipeline for per-language threshold reporting."""

from pathlib import Path
from typing import ClassVar

from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from omegaconf import DictConfig as _DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from ml_filter.data_pipelines.execution_settings import LocalExecutionSettings, SlurmExecutionSettings
from ml_filter.data_pipelines.pipeline_builder import PipelineBuilderBase

from .quantile_steps import QuantileJsonlReader


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


class QuantilePipelineBuilder(PipelineBuilderBase):
    """Builds the quantile pipeline and executor from a YAML config."""

    model_config = SettingsConfigDict(env_prefix="quantile_pipeline_", env_nested_delimiter="__")

    params: QuantilePipelineParameters
    default_job_name: ClassVar[str] = "quantile_pipeline"

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        running_on_slurm: bool | None = None,
    ) -> "QuantilePipelineBuilder":
        """Load config and build a pipeline builder instance."""
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw_config = OmegaConf.load(path)

        if "params" not in raw_config:
            raise ValueError("YAML must contain a top-level 'params:' section.")

        params_cfg = raw_config["params"]
        if isinstance(params_cfg, _DictConfig):
            params_cfg = OmegaConf.to_container(params_cfg, resolve=True)
        if not isinstance(params_cfg, dict):
            raise TypeError("`params` section must be a mapping.")

        is_running_on_slurm = (
            raw_config.get("running_on_slurm", False) if running_on_slurm is None else running_on_slurm
        )
        slurm_settings = raw_config.get("slurm_settings", None)
        local_section = raw_config.get("local_settings", None)

        if isinstance(local_section, _DictConfig):
            local_section = OmegaConf.to_container(local_section, resolve=True)
        if isinstance(slurm_settings, _DictConfig):
            slurm_settings = OmegaConf.to_container(slurm_settings, resolve=True)

        def get_param(name: str, default=None):
            return params_cfg.get(name, default)

        params = QuantilePipelineParameters(
            input_dir=get_param("input_dir"),
            glob_pattern=get_param("glob_pattern"),
            output_dir=get_param("output_dir"),
            compression=get_param("compression"),
            output_compression=get_param("output_compression", None),
            score_field=get_param("score_field", "scores"),
            selection_quantile=get_param("selection_quantile"),
            report_filename=get_param("report_filename", "quantile_report.yaml"),
            quantile_data_dir=get_param("quantile_data_dir", "quantile_data"),
        )

        builder_kwargs = {"params": params, "running_on_slurm": is_running_on_slurm}
        if is_running_on_slurm:
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
        params = self.params
        return [
            QuantileJsonlReader(
                data_folder=params.input_dir,
                glob_pattern=params.glob_pattern,
                compression=params.compression,
                score_field=params.score_field,
                selection_quantile=params.selection_quantile,
                report_path=params.report_path,
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
