"""Shared execution settings for datatrove pipeline runs."""

from __future__ import annotations

from omegaconf import DictConfig as _DictConfig
from omegaconf import OmegaConf
from pydantic import BaseModel, model_validator


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
    job_name: str = "datatrove_pipeline"
    qos: str = "normal"
    env_command: str | None = None
    condaenv: str | None = None
    venv_path: str | None = None
    # Allow users to supply any sbatch arg (e.g. nodes, ntasks, gres, account, output, error, gpus-per-task, etc.)
    # using either snake_case or dash-case. Primitive values get coerced to strings.
    sbatch_args: dict[str, str | int | float | bool] | None = None
    max_array_size: int = 1001
    depends_job_id: str | None = None
    job_id_position: int = -1
    logging_dir: str | None = None
    skip_completed: bool = True
    slurm_logs_folder: str | None = None
    max_array_launch_parallel: bool = False
    stagger_max_array_jobs: int = 0
    run_on_dependency_fail: bool = False
    randomize_start_duration: int = 0
    requeue_signals: tuple[str] | None = ("SIGUSR1",)
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
