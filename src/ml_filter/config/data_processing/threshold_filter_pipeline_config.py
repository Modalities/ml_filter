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
    input_dir: str = Field(..., description="Directory containing JSONL files to filter.")
    glob_pattern: str | None = Field(None, description="Glob pattern relative to input_dir. Optional.")
    recursive: bool = Field(True, description="Whether to search files recursively.")
    compression: str | None = Field("infer", description="Compression for JSONL inputs (infer/gzip/zstd/None).")

    score_keys: list[str] = Field(..., description="Score keys to load from each JSON line.")
    thresholds_by_score_key: dict[str, float] = Field(
        default_factory=dict,
        description="Only these keys are used for threshold filtering; others are ignored.",
    )

    id_key: str = Field("document_id", description="Key in JSON for the document id.")
    text_key: str = Field("text", description="Key in JSON for text (required by Document adapter).")

    output_dir: Path = Field(..., description="Output directory for filtered JSONL files.")
    output_filename: str = Field("${source_filename}.jsonl", description="Output filename template.")


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
