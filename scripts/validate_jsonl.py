#!/usr/bin/env python3
import json


def validate_jsonl(file_path: str) -> bool:
    """
    Validate a JSONL (JSON Lines) file.

    Args:
        file_path (str): Path to the JSONL file to validate

    Returns:
        bool: True if valid, False otherwise
    """
    try:
        # Track statistics for reporting
        total_lines = 0
        valid_lines = 0

        with open(file_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                # Skip completely empty lines
                if not line.strip():
                    continue

                total_lines += 1

                try:
                    # Attempt to parse each line as JSON
                    json.loads(line)
                    valid_lines += 1
                except json.JSONDecodeError as e:
                    # Provide detailed error information
                    print(f"JSON Decode Error on line {line_num}: {e}")
                    print(f"Problematic line: {line.strip()}")
                    return False

        # Print validation summary
        print("JSONL Validation Results:")
        print(f"Total lines processed: {total_lines}")
        print(f"Valid JSON lines: {valid_lines}")

        # Consider the file valid if at least one line was processed
        return total_lines > 0

    except IOError as e:
        print(f"Error reading file: {e}")
        return False


if __name__ == "__main__":
    validate_jsonl("data/fineweb_test/cc_annotations/0/1/cc_merged_annotations.jsonl")
