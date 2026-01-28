from functools import partial
from pathlib import Path
from typing import ClassVar, Optional

from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from omegaconf import DictConfig as _DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict

from ml_filter.data_pipelines.execution_settings import LocalExecutionSettings, SlurmExecutionSettings
from ml_filter.data_pipelines.pipeline_builder import PipelineBuilderBase

from .datatrove_jql_annotator import JQLEmbeddingReader, JQLHead, stats_adapter
from .utils import resolve_output_dtype


class AnnotationPipelineParameters(BaseModel):
    embeddings_directory: str = Field(..., description="Path to directory containing HDF5 embedding files.")
    output_keys: list[str] = Field(..., description="List of metadata keys to include in the annotated output files.")
    output_dir: Path = Field(..., description="Output directory for annotated JSONL files.")
    regression_head_checkpoints: dict[str, str] = Field(
        ..., description="Mapping of model names to head checkpoint paths."
    )
    batch_size: int = Field(..., description="Batch size for processing embeddings.")
    dataset_name: str = Field(..., description="Name of the HDF5 dataset to use.")
    compression: str | None = Field(
        ..., description="Compression for input embedding files if relevant (not used for HDF5)."
    )
    embedding_dtype: str = Field(..., description="Storage dtype for embeddings (float32, float16, bfloat16->float32).")
    label_dtype: str | None = Field(..., description="Storage dtype for labels (e.g., int8, float32). Optional.")
    model_dtype: str = Field(..., description="Model compute dtype (float32, float16, bfloat16).")

    @property
    def annotated_output_dir(self) -> Path:
        return self.output_dir / "annotated_data"


# ---------------------------------------------------------------------------
# Annotation Pipeline Builder
# ---------------------------------------------------------------------------


class AnnotationPipelineBuilder(PipelineBuilderBase):
    model_config = SettingsConfigDict(env_prefix="annotation_pipeline_", env_nested_delimiter="__")

    params: AnnotationPipelineParameters
    default_job_name: ClassVar[str] = "annotation_pipeline"
    require_local_settings: ClassVar[bool] = True
    require_slurm_settings: ClassVar[bool] = True

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

        # Helper fetch with default
        def _p(name: str, default=None):
            return params_cfg.get(name, default)

        params = AnnotationPipelineParameters(
            embeddings_directory=_p("embeddings_directory"),
            output_dir=_p("output_dir"),
            output_keys=_p("output_keys"),
            regression_head_checkpoints=_p("regression_head_checkpoints"),
            batch_size=_p("batch_size"),
            dataset_name=_p("hdf5_dataset_name"),
            compression=_p("compression"),
            embedding_dtype=_p("embedding_dtype"),
            label_dtype=_p("label_dtype"),
            model_dtype=_p("model_dtype"),
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
        # --- Unified precision validation & resolution ---
        _resolved = resolve_output_dtype(
            {
                "model_dtype": p.model_dtype,
                "embedding_dtype": p.embedding_dtype,
                "label_dtype": p.label_dtype,
            },
            pipeline="annotation_pipeline",
        )
        pipeline = [
            JQLEmbeddingReader(data_folder=p.embeddings_directory, dataset_name=p.dataset_name),
            JQLHead(
                regression_head_checkpoints=p.regression_head_checkpoints,
                batch_size=p.batch_size,
                dtype_schema={
                    "model_dtype": _resolved["model_dtype"],
                    "embedding_dtype": _resolved["embedding_dtype"],
                    "label_dtype": _resolved["label_dtype"],
                },
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
