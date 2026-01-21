"""Pydantic config schema for the threshold filtering Datatrove pipeline.

This mirrors the builder-style YAML schema used by
`ml_filter.data_processing.jsonl_filtering.threshold_filter_pipeline.ThresholdFilterPipelineBuilder`.

Intended usage:
- validate YAML structure
- provide a typed object for other orchestration code

Note: the actual execution is handled by the builder module.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field


class ThresholdFilterParamsConfig(BaseModel):
    text_input_dir: str | None = Field(None, description="Directory containing text JSONL files.")
    scores_input_dir: str | None = Field(None, description="Directory containing scores JSONL files.")
    domains_input_dir: str | None = Field(None, description="Directory containing domains JSONL files.")
    paths_file: str | None = Field(
        None,
        description="Optional file containing one relative input path per line (relative to scores_input_dir).",
    )
    glob_pattern: str | None = Field(None, description="Glob pattern relative to input_dir. Optional.")
    recursive: bool = Field(True, description="Whether to search files recursively.")
    compression: str | None = Field("infer", description="Compression for JSONL inputs (infer/gzip/zstd/None).")

    score_keys: list[str] = Field(..., description="Score keys to load from each JSON line.")
    thresholds_by_score_key: dict[str, float] = Field(
        default_factory=dict,
        description="Only these keys are used for threshold filtering; others are ignored.",
    )
    thresholds_by_folder: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description=(
            "Optional per-folder overrides for score thresholds. Keys are top-level folder names "
            "(e.g., Deu_Latn) and values are per-score thresholds for files under that folder."
        ),
    )

    text_jsonl_id_key: str = Field("document_id", description="Key in TEXT JSONL for the document id.")
    score_jsonl_id_key: str = Field("document_id", description="Key in SCORES JSONL for the document id.")
    domain_jsonl_id_key: str = Field("document_id", description="Key in DOMAINS JSONL for the document id.")
    domain_jsonl_domain_key: str = Field("domain", description="Key in DOMAINS JSONL for the domain string.")
    text_jsonl_text_key: str = Field("text", description="Key in TEXT JSONL for the text content.")
    accepted_domains: list[str] = Field(
        default_factory=list,
        description="Accepted domain values when domains_input_dir is provided.",
    )

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

    min_num_words: int | None = Field(
        None,
        description="If set, keep only lines whose word-count (in num_words_column) is >= this value.",
    )
    num_words_column: str = Field(
        "text",
        description="JSON key to compute word-count from when min_num_words is provided.",
    )

    output_dir: Path = Field(..., description="Output directory for filtered JSONL files.")
    output_filename: str = Field("${file_relpath}", description="Output filename template.")


class ThresholdFilterLocalSettingsConfig(BaseModel):
    tasks: int = 1
    local_tasks: int = 1
    local_rank_offset: int = 0
    workers: int = -1
    logging_dir: str | None = None


class ThresholdFilterSlurmSettingsConfig(BaseModel):
    tasks: int = 1
    time: str = "00:30:00"
    partition: str = "default"
    cpus_per_task: int = 4
    mem_per_cpu_gb: int = 8
    workers: int = -1
    job_name: str = "threshold_filter_pipeline"
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


class ThresholdFilterPipelineConfig(BaseModel):
    """Builder-style config.

    YAML layout:
    
    running_on_slurm: bool
    params: {...}
    local_settings: {...}  # only when running_on_slurm=false
    slurm_settings: {...}  # only when running_on_slurm=true
    """

    running_on_slurm: bool = False
    params: ThresholdFilterParamsConfig
    local_settings: ThresholdFilterLocalSettingsConfig | None = None
    slurm_settings: ThresholdFilterSlurmSettingsConfig | None = None
