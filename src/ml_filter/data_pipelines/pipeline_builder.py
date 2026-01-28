"""Shared pipeline builder helpers for datatrove runs."""

from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ml_filter.data_pipelines.execution_settings import LocalExecutionSettings, SlurmExecutionSettings


class PipelineBuilderBase(BaseSettings):
    """Base class that validates execution mode and fills shared defaults."""

    model_config = SettingsConfigDict(env_nested_delimiter="__")

    params: BaseModel
    running_on_slurm: bool = False
    local_settings: LocalExecutionSettings | None = None
    slurm_settings: SlurmExecutionSettings | None = None

    default_job_name: ClassVar[str] = "datatrove_pipeline"
    require_local_settings: ClassVar[bool] = False
    require_slurm_settings: ClassVar[bool] = False

    @model_validator(mode="after")
    def slurm_vs_local(self):
        if self.running_on_slurm and self.local_settings is not None:
            raise ValueError("Running on Slurm requires slurm execution settings, not local settings.")
        if self.running_on_slurm and self.slurm_settings is None:
            if self.require_slurm_settings:
                raise ValueError("running_on_slurm=True requires 'slurm_settings' section.")
            self.slurm_settings = SlurmExecutionSettings()
        elif not self.running_on_slurm and self.slurm_settings is not None:
            raise ValueError("Running locally requires local execution settings, not Slurm settings.")
        if not self.running_on_slurm and self.local_settings is None:
            if self.require_local_settings:
                raise ValueError("running_on_slurm=False requires 'local_settings' section.")
            self.local_settings = LocalExecutionSettings()
        return self

    @model_validator(mode="after")
    def set_default_job_name(self):
        if self.slurm_settings is not None and "job_name" not in self.slurm_settings.model_fields_set:
            self.slurm_settings.job_name = self.default_job_name
        return self

    @model_validator(mode="after")
    def set_logging_dir(self):
        output_dir = getattr(self.params, "output_dir", None)
        if output_dir is None:
            return self
        output_dir = Path(output_dir)
        if self.local_settings is not None and self.local_settings.logging_dir is None:
            self.local_settings.logging_dir = str(output_dir / "logs")
        if self.slurm_settings is not None and self.slurm_settings.logging_dir is None:
            self.slurm_settings.logging_dir = str(output_dir / "logs")
        return self
