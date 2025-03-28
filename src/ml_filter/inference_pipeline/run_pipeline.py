from pathlib import Path

import torch
import yaml
from pydantic import ValidationError

from ml_filter.inference_pipeline.config import Config
from ml_filter.inference_pipeline.inference import InferencePipeline
from ml_filter.inference_pipeline.logging import get_logger
from ml_filter.inference_pipeline.model_factory import ModelFactory


def load_config(config_path: Path) -> Config:
    with open(config_path, "r") as file:
        config_data = yaml.safe_load(file)

    try:
        return Config(**config_data)
    except ValidationError as e:
        print("Configuration validation failed:", e)
        exit(1)


def run_pipeline(config_file_path: Path):
    config = load_config(config_file_path)
    print("Loaded configuration:", config)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger = get_logger(logger_id="inference_pipeline", logging_dir_path=config.paths.logging_dir)
    model = ModelFactory.load_huggingface_model_checkpoint(
        config.model_settings.model_checkpoint_path,
        config.model_settings.model_type,
        config.model_settings.num_regressor_outputs,
        config.model_settings.num_classes_per_output,
        config.model_settings.use_regression,
        device,
        logger,
    )

    inference_pipeline = InferencePipeline(
        model=model,
        sequence_length=config.model_settings.sequence_length,
        device=device,
        logger=logger,
        prediction_key=config.pipeline_settings.prediction_key,
        batch_size=config.pipeline_settings.batch_size,
        output_dir=config.paths.output_dir,
        input_files_list_path=config.paths.input_files_list_path,
        processed_files_list_path=config.paths.processed_files_list_path,
    )
    inference_pipeline.run()
