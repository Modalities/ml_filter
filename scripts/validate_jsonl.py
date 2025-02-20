import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def read_jsonl_file(file_path: Path) -> list:
    """Reads a JSONL file and returns its lines.

    Args:
        file_path (Path): The path to the JSONL file.

    Returns:
        list: A list of non-empty lines from the file.
    """
    try:
        with file_path.open("r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]  # Skip empty lines
    except IOError as e:
        logging.error(f"Error reading file: {e}")
        return []


def is_valid_json_line(line: str, line_num: int) -> bool:
    """Validates a single JSON line.

    Args:
        line (str): The JSON line to validate.
        line_num (int): The line number in the file.

    Returns:
        bool: True if the line is valid JSON, False otherwise.
    """
    try:
        json.loads(line)
        return True
    except json.JSONDecodeError as e:
        logging.warning(f"Invalid JSON on line {line_num}: {e}")
        logging.warning(f"Problematic line: {line}")
        return False


def validate_jsonl(file_path: Path) -> dict:
    """Validates a JSONL (JSON Lines) file.

    Args:
        file_path (Path): A Path object representing the JSONL file.

    Returns:
        dict: A summary of the validation results with:
            - `valid` (int): Number of valid JSON lines.
            - `invalid` (int): Number of invalid JSON lines.
            - `total` (int): Total number of lines processed.
    """
    if not file_path.exists():
        logging.error(f"File not found: {file_path}")
        return {"valid": 0, "invalid": 0, "total": 0, "status": "File not found"}

    if not file_path.is_file():
        logging.error(f"Not a valid file: {file_path}")
        return {"valid": 0, "invalid": 0, "total": 0}

    lines = read_jsonl_file(file_path)

    valid_lines = sum(1 for i, line in enumerate(lines) if is_valid_json_line(line, i))
    total_lines = len(lines)
    invalid_lines = total_lines - valid_lines

    logging.info("JSONL Validation Summary:")
    logging.info(f"Total lines processed: {total_lines}")
    logging.info(f"Valid JSON lines: {valid_lines}")
    logging.info(f"Invalid JSON lines: {invalid_lines}")

    return {
        "valid": valid_lines,
        "invalid": invalid_lines,
        "total": total_lines,
    }
