import logging
from pathlib import Path


def get_logger(logger_id: str, logging_dir_path: Path) -> logging.Logger:
    logging_dir_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        filename=logging_dir_path / f"logs_{logger_id}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    return logging.getLogger(f"logger_{logger_id}")
