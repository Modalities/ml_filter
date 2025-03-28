import logging
from pathlib import Path


def get_logger(logger_id: str, logging_dir_path: Path) -> logging.Logger:
    logging_dir_path.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"logger_{logger_id}")
    logger.setLevel(logging.INFO)

    # Prevent duplicate log entries if logger already has handlers
    if not logger.hasHandlers():
        # File Handler
        file_handler = logging.FileHandler(logging_dir_path / f"logs_{logger_id}.log")
        file_handler.setLevel(logging.INFO)

        # Console (Stdout) Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        # Log format
        log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
