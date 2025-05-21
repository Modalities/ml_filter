import logging
import os
from datetime import datetime

from transformers import logging as hf_logging


def setup_logging(log_dir="outputs/logs"):
    os.makedirs(log_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"training_{timestamp}.log")

    # Root logger (for your own app code)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers

    # File handler for root logger
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s — %(levelname)s — %(name)s — %(message)s")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)

    # Console handler for root logger
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter("%(levelname)s — %(message)s")
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Setup Hugging Face logging (keep default console handler)
    hf_logging.set_verbosity_info()
    hf_logging.enable_default_handler()
    hf_logging.enable_explicit_format()

    # Add the same file handler to Hugging Face logger so it writes to file too
    hf_logger = logging.getLogger("transformers")
    hf_logger.addHandler(file_handler)

    # Also set HF logger level (optional, usually INFO)
    hf_logger.setLevel(logging.INFO)

    # Similarly for datasets logger
    datasets_logger = logging.getLogger("datasets")
    datasets_logger.addHandler(file_handler)
    datasets_logger.setLevel(logging.INFO)

    logger.info(f"Logging is set up. Log file: {log_file}")
    return logger

