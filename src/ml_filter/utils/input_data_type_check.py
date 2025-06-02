import os
from pathlib import Path
from omegaconf import DictConfig

def is_valid_path(dataset_path: str | Path, dataset_type: str) -> bool:
    if os.path.isfile(dataset_path) and dataset_path.endswith(f".{dataset_type}"):
        return True
    if os.path.isdir(dataset_path):
        return any(
            f.endswith(f".{dataset_type}") and os.path.isfile(os.path.join(dataset_path, f))
            for f in os.listdir(dataset_path)
        )
    return False


def check_datatype_consistency(cfg: DictConfig, dataset_type: str):
    if not (is_valid_path(dataset_path=cfg.data.train_file_path, dataset_type=dataset_type) and is_valid_path(
            cfg.data.val_file_path, dataset_type=dataset_type)):
        raise ValueError(
            f"Invalid dataset paths. Please provide valid {dataset_type} file paths or directories containing {dataset_type} files."
        )