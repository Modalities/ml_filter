"""Average-threshold filtering pipeline utilities.

This module wires `PairedAverageThresholdFilter` into a Datatrove pipeline
followed by a `JsonlWriter`. It supports both local execution and Slurm execution
via the respective Datatrove executors.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from datatrove.executor import LocalPipelineExecutor, SlurmPipelineExecutor
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers import JsonlWriter
from omegaconf import OmegaConf
from omegaconf import DictConfig as _DictConfig
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import SettingsConfigDict

from ml_filter.data_pipelines.execution_settings import LocalExecutionSettings, SlurmExecutionSettings
from ml_filter.data_pipelines.filtering.numwords_filter import NumWordsFilter
from ml_filter.data_pipelines.filtering.paired_average_threshold_filter import PairedAverageThresholdFilter
from ml_filter.data_pipelines.pipeline_builder import PipelineBuilderBase


def _jsonl_writer_adapter(self, document):
    """Return id/text plus configured score/domain fields for output."""
    data = {
        "id": getattr(document, "id", None),
        "text": getattr(document, "text", None),
    }
    metadata = getattr(document, "metadata", None)
    if isinstance(metadata, dict):
        for key in getattr(self, "score_keys", []):
            if key in metadata:
                data[key] = metadata[key]
        domain_key = getattr(self, "domain_key", None)
        if domain_key and domain_key in metadata:
            data[domain_key] = metadata[domain_key]
    return data


class AverageThresholdFilterPipelineParameters(BaseModel):
    text_input_dir: str | None = Field(None, description="Directory containing text JSONL files.")
    scores_input_dir: str | None = Field(None, description="Directory containing scores JSONL files.")
    domains_input_dir: str | None = Field(None, description="Directory containing domains JSONL files.")
    glob_pattern: str | None = Field(None, description="Glob pattern relative to input_dir. Optional.")
    recursive: bool = Field(True, description="Whether to search files recursively.")
    compression: str | None = Field("infer", description="Compression for JSONL inputs (infer/gzip/zstd/None).")

    # Filtering
    score_keys: list[str] = Field(..., description="Score keys to load from each JSON line.")
    average_threshold: float | None = Field(
        None, description="Average threshold applied across all score keys."
    )
    average_thresholds_by_folder: dict[str, float] = Field(
        default_factory=dict,
        description=(
            "Optional per-folder overrides for the averaged threshold. Keys are top-level folder names "
            "(e.g., Deu_Latn) and values are average thresholds for files under that folder."
        ),
    )

    # Paired-mode id keys (explicit)
    text_jsonl_id_key: str = Field(
        "document_id",
        description="Key in TEXT JSONL used to validate alignment with scores JSONL.",
    )
    score_jsonl_id_key: str = Field(
        "document_id",
        description="Key in SCORES JSONL used to validate alignment with text JSONL.",
    )
    domain_jsonl_id_key: str = Field(
        "document_id",
        description="Key in DOMAINS JSONL used to validate alignment with text JSONL.",
    )
    domain_jsonl_domain_key: str = Field(
        "domain",
        description="Key in DOMAINS JSONL that contains the domain string.",
    )
    text_jsonl_text_key: str = Field(
        "text",
        description="Key in TEXT JSONL that contains the actual text content.",
    )
    accepted_domains: list[str] = Field(
        default_factory=list,
        description="Accepted domain values when domains_input_dir is provided.",
    )

    # Paired-mode alignment error handling
    on_mismatch: str = Field(
        "raise",
        description="Behavior when ids mismatch in paired files: raise|skip_line|skip_file.",
    )
    max_mismatches_per_file: int = Field(
        0,
        description=(
            "Safety limit for id mismatches per file when on_mismatch != 'raise'. "
            "0 means unlimited."
        ),
    )
    # Backward compatible alias (if provided in YAML, we map it to both keys)
    document_id_key: str | None = Field(
        None,
        description="(Deprecated) If set, used as both text_jsonl_id_key and score_jsonl_id_key.",
    )

    # Optional word-count filter (applied after threshold filter)
    min_num_words: int | None = Field(
        None,
        description="If set, additionally keep only lines whose word-count (in num_words_column) is >= this value.",
    )
    num_words_column: str = Field(
        "text",
        description="JSON key to compute word-count from when min_num_words is provided.",
    )

    # Writer
    output_dir: Path = Field(..., description="Output directory for filtered JSONL files.")
    output_filename: str = Field(
        "${file_relpath}",
        description=(
            "Output filename template. Datatrove's JsonlWriter uses string.Template variables. "
            "Use ${file_relpath} to preserve the input folder hierarchy (requires readers to provide file_relpath in metadata), "
            "or ${file_stem}.jsonl to write into a flat output directory."
        ),
    )


class AverageThresholdFilterPipelineBuilder(PipelineBuilderBase):
    model_config = SettingsConfigDict(env_prefix="average_threshold_filter_pipeline_", env_nested_delimiter="__")

    params: AverageThresholdFilterPipelineParameters
    running_on_slurm: bool = False
    local_settings: LocalExecutionSettings | None = None
    slurm_settings: SlurmExecutionSettings | None = None

    @model_validator(mode="after")
    def _validate_execution_mode(self):
        if self.running_on_slurm:
            if self.local_settings is not None:
                raise ValueError("Running on Slurm requires only slurm_settings, not local_settings.")
            if self.slurm_settings is None:
                self.slurm_settings = SlurmExecutionSettings()
        else:
            if self.slurm_settings is not None:
                raise ValueError("Running locally requires only local_settings, not slurm_settings.")
            if self.local_settings is None:
                self.local_settings = LocalExecutionSettings()
        return self

    @model_validator(mode="after")
    def _set_logging_dir(self):
        if self.local_settings and self.local_settings.logging_dir is None:
            self.local_settings.logging_dir = str(self.params.output_dir / "logs")
        if self.slurm_settings and self.slurm_settings.logging_dir is None:
            self.slurm_settings.logging_dir = str(self.params.output_dir / "logs")
        return self

    @classmethod
    def from_yaml(
        cls, path: Path, running_on_slurm: Optional[bool] = None
    ) -> "AverageThresholdFilterPipelineBuilder":
        if not path.is_file():
            raise FileNotFoundError(f"Config file not found: {path}")
        raw = OmegaConf.load(path)

        if "params" not in raw:
            raise ValueError("YAML must contain a top-level 'params:' section (builder-style schema).")

        params_cfg = raw["params"]
        # Convert to a plain dict without resolving interpolations like "${source_filename}".
        if isinstance(params_cfg, _DictConfig):
            params_cfg = OmegaConf.to_container(params_cfg, resolve=False)  # type: ignore[assignment]

        rs = raw.get("running_on_slurm") if running_on_slurm is None else running_on_slurm
        if rs is None:
            raise ValueError("YAML must specify 'running_on_slurm'.")

        local_section = raw.get("local_settings", None)
        slurm_section = raw.get("slurm_settings", None)

        def _p(name: str, default=None):
            # params_cfg is a plain dict at this point
            return params_cfg.get(name, default)

        params = AverageThresholdFilterPipelineParameters(
            text_input_dir=_p("text_input_dir", None),
            scores_input_dir=_p("scores_input_dir", None),
            domains_input_dir=_p("domains_input_dir", None),
            glob_pattern=_p("glob_pattern"),
            recursive=_p("recursive", True),
            compression=_p("compression", "infer"),
            score_keys=list(_p("score_keys")),
            average_threshold=_p("average_threshold", None),
            average_thresholds_by_folder=dict(_p("average_thresholds_by_folder", {})),
            text_jsonl_id_key=_p("text_jsonl_id_key", _p("document_id_key", "document_id")),
            score_jsonl_id_key=_p("score_jsonl_id_key", _p("document_id_key", "document_id")),
            domain_jsonl_id_key=_p("domain_jsonl_id_key", _p("document_id_key", "document_id")),
            domain_jsonl_domain_key=_p("domain_jsonl_domain_key", "domain"),
            text_jsonl_text_key=_p("text_jsonl_text_key", "text"),
            accepted_domains=list(_p("accepted_domains", []) or []),
            document_id_key=_p("document_id_key", None),
            on_mismatch=_p("on_mismatch", "raise"),
            max_mismatches_per_file=int(_p("max_mismatches_per_file", 0)),
            min_num_words=_p("min_num_words", None),
            num_words_column=_p("num_words_column", "text"),
            output_dir=Path(_p("output_dir")),
            output_filename=_p("output_filename", "${file_relpath}"),
        )

        builder_kwargs: dict = {"params": params, "running_on_slurm": rs}
        if rs:
            if slurm_section is not None:
                builder_kwargs["slurm_settings"] = SlurmExecutionSettings(**slurm_section)
        else:
            if local_section is not None:
                if not isinstance(local_section, _DictConfig):
                    local_section = OmegaConf.create(local_section)
                builder_kwargs["local_settings"] = LocalExecutionSettings(**local_section)

        return cls(**builder_kwargs)

    def build_pipeline(self) -> list[PipelineStep]:
        p = self.params
        if not (p.text_input_dir and p.scores_input_dir):
            raise ValueError(
                "This pipeline supports only paired filtering; provide both text_input_dir and scores_input_dir."
            )
        reader: PipelineStep = PairedAverageThresholdFilter(
            text_data_folder=p.text_input_dir,
            scores_data_folder=p.scores_input_dir,
            score_keys=p.score_keys,
            average_threshold=p.average_threshold,
            average_thresholds_by_folder=p.average_thresholds_by_folder,
            text_jsonl_id_key=p.text_jsonl_id_key,
            score_jsonl_id_key=p.score_jsonl_id_key,
            text_jsonl_text_key=p.text_jsonl_text_key,
            domains_data_folder=p.domains_input_dir,
            accepted_domains=p.accepted_domains,
            domain_jsonl_id_key=p.domain_jsonl_id_key,
            domain_jsonl_domain_key=p.domain_jsonl_domain_key,
            compression=p.compression,  # type: ignore[arg-type]
            recursive=p.recursive,
            glob_pattern=p.glob_pattern,
            on_mismatch=p.on_mismatch,  # type: ignore[arg-type]
            max_mismatches_per_file=p.max_mismatches_per_file,
        )

        pipeline: list[PipelineStep] = [reader]

        if p.domains_input_dir and not p.accepted_domains:
            raise ValueError("accepted_domains must be provided when domains_input_dir is set.")

        if p.min_num_words is not None:
            pipeline.append(
                NumWordsFilter(
                    min_num_words=p.min_num_words,
                    column=p.num_words_column,
                )
            )

        writer = JsonlWriter(
            output_folder=str(p.output_dir),
            output_filename=p.output_filename,
            expand_metadata=False,
            adapter=_jsonl_writer_adapter,
            compression=None,
        )
        writer.score_keys = p.score_keys
        writer.domain_key = p.domain_jsonl_domain_key if p.domains_input_dir else None
        pipeline.append(writer)
        return pipeline

    def build_executor(self) -> LocalPipelineExecutor | SlurmPipelineExecutor:
        pipeline = self.build_pipeline()
        if self.running_on_slurm:
            return SlurmPipelineExecutor(pipeline=pipeline, **self.slurm_settings.model_dump())  # type: ignore[union-attr]
        return LocalPipelineExecutor(pipeline=pipeline, **self.local_settings.model_dump())  # type: ignore[union-attr]


def run_average_threshold_filter_pipeline(config: Path | AverageThresholdFilterPipelineBuilder):
    """Run the average-threshold filter pipeline.

    Args:
        config: Either a Path to a YAML config file (builder-style schema) or an
            already constructed AverageThresholdFilterPipelineBuilder.
    """
    builder = (
        config
        if isinstance(config, AverageThresholdFilterPipelineBuilder)
        else AverageThresholdFilterPipelineBuilder.from_yaml(config)
    )
    executor = builder.build_executor()
    executor.run()


def _main(argv: list[str] | None = None) -> int:
    """Module CLI.

    Usage:
        python -m ml_filter.data_processing.jsonl_filtering.average_threshold_filter_pipeline <config.yaml>
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run average threshold filter pipeline from a YAML config")
    parser.add_argument("config", type=Path, help="Path to a builder-style YAML config")
    args = parser.parse_args(argv)

    run_average_threshold_filter_pipeline(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
