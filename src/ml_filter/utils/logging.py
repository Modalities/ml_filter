
import logging
from typing import Optional


def get_logger(name: str, level: Optional[int] = logging.INFO, format: Optional[str] = None) -> logging.Logger:
    """Create a logger instance with the specified name.

    Args:
        name (str): The name of the logger.
        level (int, optional): The logging level. Defaults to logging.INFO.
        format (Optional[str], optional): The log format. Defaults to None.

    Returns:
        logging.Logger: The logger instance.
    """
    logging_args = {"level": level}
    if format:
        logging_args["format"] = format
    logging.basicConfig(**logging_args)
    return logging.getLogger(name)
