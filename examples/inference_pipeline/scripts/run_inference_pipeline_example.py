import os
from pathlib import Path

from ml_filter.inference_pipeline.run_pipeline import run_pipeline

script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
config_file_path = script_dir / "../configs/inference_pipeline_config.yaml"  # Construct relative path
config_file_path = config_file_path.resolve()

run_pipeline(config_file_path)
