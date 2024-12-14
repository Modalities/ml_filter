import json

# Configure logging
import logging
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

import jq

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def compute_num_words_and_chars_in_jsonl(input_file_path: Path, output_file_path: Path) -> None:
    """Processes a JSONL file to count words in each document and writes the results to a JSON file.

    Args:
        input_file (str): The path to the input JSONL file.
        output_file (str): The path to the output JSON file.
    """
    word_count_to_doc_count = Counter()
    total_word_count, total_char_count = 0, 0

    # Open the JSONL file and process line by line
    with input_file_path.open("r", encoding="utf-8") as file:
        for line in file:
            document = json.loads(line.strip())
            if "text" in document:
                text = document["text"]
                words = text.split()
                word_count = len(words)
                char_count = len(text)
                word_count_to_doc_count[word_count] += 1

                total_word_count += word_count
                total_char_count += char_count

    # Prepare the output dictionary
    output_data = {
        "word_counts": dict(word_count_to_doc_count),
        "total_num_words": total_word_count,
        "total_num_chars": total_char_count,
    }

    # Write the result to a JSON file
    with output_file_path.open("w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4)


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _count_words_in_file(file_path: Path) -> tuple[Path, int]:
    """Reads a JSONL file using the jq module, extracts the 'text' field from each line,
    and counts the total words.

    Args:
        file_path (Path): Path to the JSONL file.

    Returns:
        tuple[Path, int]: A tuple containing the file path and the total word count.
    """
    word_count = 0
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    # Use jq to extract the 'text' field
                    text = jq.compile(".text").input(text=line).first()
                    if text:
                        word_count += len(text.split())
                except Exception as e:
                    logger.warning(f"Skipping line in {file_path} due to error: {e}")
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
    return file_path, word_count


def find_jsonl_files(directory_path: Path) -> list[Path]:
    """
    Recursively finds all JSONL files in a directory and its subdirectories.

    Args:
        directory_path (Path): Path to the root directory to search.

    Returns:
        list[Path]: A list of paths to JSONL files.
    """
    return list(Path(directory_path).rglob("*.jsonl"))


def start_word_count_jsonl_files(directory: Path, output_file: Path) -> None:
    """
    Processes all JSONL files in a directory to count words and saves the results to an output file.

    Args:
        directory (Path): Path to the directory containing JSONL files.
        output_file (Path): Path to the output file (JSONL) to save results.
    """
    files = find_jsonl_files(directory)
    if not files:
        logger.info("No JSONL files found.")
        return

    logger.info(f"Found {len(files)} JSONL files. Processing...")

    with Pool(processes=cpu_count()) as pool:
        results = pool.map(count_words_in_file, files)

    # Save results as a JSONL or YAML file
    output_data: dict[Path, int] = {file_path: word_count for file_path, word_count in results}
    with open(output_file, "w", encoding="utf-8") as f:
        for file_path, word_count in output_data.items():
            f.write(json.dumps({str(file_path): word_count}) + "\n")

    logger.info(f"Word counts saved to {output_file}")
