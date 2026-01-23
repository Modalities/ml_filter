import logging
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

from .datatrove_jql_annotator import HDF5Writer, JQLEmbedder, JQLJsonlReader
from .utils import resolve_output_dtype

logger = logging.getLogger(__name__)


class EmbeddingPipelineParameters(BaseModel):
    input_dir: str = Field(..., description="Directory containing JSONL files.")
    glob_pattern: str = Field(..., description="Glob for selecting JSONL files.")
    keys_to_index: list[str] = Field(..., description="List of keys to index in the output HDF5.")
    text_field: str = Field(..., description="Key name in JSON for the raw text field to embed.")
    compression: str | None = Field(..., description="Compression for input JSONL files (infer/gzip/zstd/None).")
    embedding_dtype: str = Field(
        ..., description="Storage dtype for embeddings (float32, float16, bfloat16->float32 storage)."
    )
    label_dtype: str = Field(
        ..., description="Storage dtype for labels (e.g., int8, float32). Optional if labels disabled."
    )
    model_dtype: str = Field(..., description="Model compute dtype (float32, float16, bfloat16).")
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


class EmbeddingPipelineBuilder(PipelineBuilderBase):
    model_config = SettingsConfigDict(env_prefix="embedding_pipeline_", env_nested_delimiter="__")

    params: EmbeddingPipelineParameters
    default_job_name: ClassVar[str] = "embedding_pipeline"

    @classmethod
    def from_yaml(
        cls,
        path: Path,
        running_on_slurm: bool | None = None,
    ) -> "EmbeddingPipelineBuilder":
        """Create a builder directly from a YAML file.

        Args:
            path: Path to YAML file
            running_on_slurm: Optional override execution mode. If None, value will be read from YAML
                (key: `running_on_slurm`; defaults to False when absent). If a boolean is provided it
                takes precedence over the YAML value.
        """
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = OmegaConf.load(path)

        if "params" not in raw:
            raise ValueError("YAML must contain a top-level 'params:' section (builder-style schema).")

        params_cfg = raw["params"]
        if isinstance(params_cfg, _DictConfig):
            OmegaConf.resolve(params_cfg)
            params_cfg = OmegaConf.to_container(params_cfg, resolve=True)  # type: ignore[assignment]
        if not isinstance(params_cfg, dict):
            raise TypeError("`params` section must be a mapping.")

        # Respect explicit override only when the caller supplies a value; otherwise honor YAML
        rs = raw.get("running_on_slurm", False) if running_on_slurm is None else running_on_slurm
        slurm_settings = raw.get("slurm_settings", None)
        local_section = raw.get("local_settings", None)

        if isinstance(local_section, _DictConfig):
            local_section = OmegaConf.to_container(local_section, resolve=True)
        if isinstance(slurm_settings, _DictConfig):
            slurm_settings = OmegaConf.to_container(slurm_settings, resolve=True)
        if local_section is not None and not isinstance(local_section, dict):
            raise TypeError("`local_settings` section must be a mapping when provided.")
        if slurm_settings is not None and not isinstance(slurm_settings, dict):
            raise TypeError("`slurm_settings` section must be a mapping when provided.")

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
            glob_pattern=_p("glob_pattern"),
            keys_to_index=_p("keys_to_index"),
            text_field=_p("text_field"),
            compression=_p("compression"),
            embedding_dtype=_p("embedding_dtype"),
            label_dtype=_p("label_dtype"),
            model_dtype=_p("model_dtype"),
            output_dir=_p("output_dir"),
            embedding_dir=_p("embedding_dir"),
            embedding_model=_p("embedding_model"),
            hdf5_dataset_name=_p("hdf5_dataset_name", "train"),
            batch_size=_p("batch_size"),
            writer_batch_size=_p("writer_batch_size"),
            max_length=_p("max_length"),
            padding=_p("padding"),
            truncation=_p("truncation"),
            save_labels=_p("save_labels"),
        )
        builder_kwargs = {"params": params, "running_on_slurm": rs}

        # Unified execution settings parsing
        if rs:
            if slurm_settings is not None:
                # Convert DictConfig to plain dict first
                if isinstance(slurm_settings, _DictConfig):
                    slurm_settings = OmegaConf.to_container(slurm_settings, resolve=True)
                builder_kwargs["slurm_settings"] = SlurmExecutionSettings(**slurm_settings)
        else:
            if isinstance(local_section, _DictConfig):
                local_section = OmegaConf.to_container(local_section, resolve=True)
            if isinstance(local_section, dict):
                builder_kwargs["local_settings"] = LocalExecutionSettings(
                    **{k: v for k, v in local_section.items() if k in LocalExecutionSettings.model_fields}
                )

        return cls(**builder_kwargs)

    def build_pipeline(self) -> list[PipelineStep]:
        p = self.params
        # --- Unified precision validation & resolution ---
        _resolved = resolve_output_dtype(
            {
                "model_dtype": p.model_dtype,
                "embedding_dtype": p.embedding_dtype,
                "label_dtype": p.label_dtype,
            },
            pipeline="embedding_pipeline",
        )
        pipeline: list[PipelineStep] = [
            JQLJsonlReader(
                data_folder=p.input_dir,
                keys_to_index=p.keys_to_index,
                glob_pattern=p.glob_pattern,
                text_key=p.text_field,
                save_labels=p.save_labels,
            ),
            JQLEmbedder(
                embedder_model_id=p.embedding_model,
                batch_size=p.batch_size,
                max_length=p.max_length,
                padding=p.padding,
                truncation=p.truncation,
                model_dtype=_resolved["model_dtype"],
                stats_writer=HDF5Writer(
                    output_folder=str(p.embedding_output_dir),
                    output_filename="${source_filename}.h5",
                    dataset_name=p.hdf5_dataset_name,
                    batch_size=p.writer_batch_size,
                    save_labels=p.save_labels,
                    compression=p.compression,
                    dtype_schema={
                        "embedding_dtype": _resolved["embedding_dtype"],
                        "label_dtype": _resolved["label_dtype"],
                    },
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
