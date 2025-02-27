from pathlib import Path

from pydantic import BaseModel


class EnvironmentConfig(BaseModel):
    task_id: int


class PathsConfig(BaseModel):
    input_files_list_path: Path
    processed_files_list_path: Path
    output_dir: Path
    logging_dir: Path


class ModelSettingsConfig(BaseModel):
    model_checkpoint_path: Path | str
    model_type: str
    num_regressor_outputs: int
    num_classes_per_output: list[int]
    use_regression: bool
    sequence_length: int


class PipelineSettingsConfig(BaseModel):
    batch_size: int
    prediction_key: str


class Config(BaseModel):
    environment: EnvironmentConfig
    paths: PathsConfig
    model_settings: ModelSettingsConfig
    pipeline_settings: PipelineSettingsConfig
