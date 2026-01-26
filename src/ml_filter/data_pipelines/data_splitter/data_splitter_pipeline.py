"""Config-driven data splitter pipeline for bucketed JSONL output."""

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

from .splitter_steps import BucketedJsonlSplitter


class DataSplitterPipelineParameters(BaseModel):
    """Configuration parameters for the data splitter pipeline."""

    input_dir: str = Field(..., description="Directory containing JSONL files.")
    glob_pattern: str = Field(..., description="Glob for selecting JSONL files.")
    output_dir: Path = Field(..., description="Root output directory.")
    compression: str | None = Field(..., description="Compression for input JSONL files (infer/gzip/zstd/None).")
    score_fields: list[str] = Field(
        ...,
        description="Score fields to average per document (e.g. ['score_llama', 'score_mistral']).",
    )
    buckets: list[tuple[float, float]] = Field(
        ...,
        description="Score buckets as [lower, upper] pairs.",
    )
    output_subdir: str = Field("data_buckets", description="Bucket output subdirectory under each language.")
    bucket_filename_template: str = Field(
        "{lower}_{upper}_bucked_{language}.jsonl",
        description="Template for bucket filenames. Uses {lower}, {upper}, {language}.",
    )


class DataSplitterPipelineBuilder(PipelineBuilderBase):
    """Builds the data splitter pipeline and executor from a YAML config."""

    model_config = SettingsConfigDict(env_prefix="data_splitter_pipeline_", env_nested_delimiter="__")

    params: DataSplitterPipelineParameters
    default_job_name: ClassVar[str] = "data_splitter_pipeline"

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        running_on_slurm: bool | None = None,
    ) -> "DataSplitterPipelineBuilder":
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

        score_fields = get_param("score_fields", None)
        if not isinstance(score_fields, list) or not all(isinstance(field, str) for field in score_fields):
            raise TypeError("score_fields must be a list of strings.")

        raw_buckets = get_param("buckets", None)
        if not isinstance(raw_buckets, list) or len(raw_buckets) == 0:
            raise TypeError("buckets must be a non-empty list of [lower, upper] pairs.")
        buckets: list[tuple[float, float]] = []
        for bucket in raw_buckets:
            if isinstance(bucket, dict):
                if "lower" not in bucket or "upper" not in bucket:
                    raise ValueError("Bucket dicts must include 'lower' and 'upper' keys.")
                lower = bucket["lower"]
                upper = bucket["upper"]
            else:
                if not isinstance(bucket, (list, tuple)) or len(bucket) != 2:
                    raise ValueError("Each bucket must be a [lower, upper] pair.")
                lower, upper = bucket
            buckets.append((float(lower), float(upper)))

        params = DataSplitterPipelineParameters(
            input_dir=get_param("input_dir"),
            glob_pattern=get_param("glob_pattern"),
            output_dir=get_param("output_dir"),
            compression=get_param("compression"),
            score_fields=score_fields,
            buckets=buckets,
            output_subdir=get_param("output_subdir", "data_buckets"),
            bucket_filename_template=get_param(
                "bucket_filename_template",
                "{lower}_{upper}_bucked_{language}.jsonl",
            ),
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
            BucketedJsonlSplitter(
                data_folder=params.input_dir,
                glob_pattern=params.glob_pattern,
                compression=params.compression,
                score_fields=params.score_fields,
                buckets=params.buckets,
                output_dir=params.output_dir,
                output_subdir=params.output_subdir,
                bucket_filename_template=params.bucket_filename_template,
            ),
        ]

    def build_executor(self) -> LocalPipelineExecutor | SlurmPipelineExecutor:
        """Create the local or Slurm executor based on config."""
        pipeline = self.build_pipeline()
        if self.running_on_slurm:
            print("Running Slurm Data Splitter Pipeline Executor")
            return SlurmPipelineExecutor(pipeline=pipeline, **self.slurm_settings.model_dump())
        print("Running Local Data Splitter Pipeline Executor")
        return LocalPipelineExecutor(pipeline=pipeline, **self.local_settings.model_dump())


def run_data_splitter_pipeline(config_file_path: Path):
    """Run data splitter pipeline directly from YAML file."""
    builder = DataSplitterPipelineBuilder.from_yaml(config_file_path)
    executor = builder.build_executor()
    executor.run()
