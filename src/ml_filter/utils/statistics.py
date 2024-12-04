import json
from collections import Counter
from pathlib import Path


def compute_num_words_in_jsonl(input_file_path: Path, output_file_path: Path) -> None:
    """Processes a JSONL file to count words in each document and writes the results to a JSON file.

    Args:
        input_file (str): The path to the input JSONL file.
        output_file (str): The path to the output JSON file.
    """
    word_count_to_doc_count = Counter()
    total_word_count = 0

    # Open the JSONL file and process line by line
    with input_file_path.open("r", encoding="utf-8") as file:
        for line in file:
            document = json.loads(line.strip())
            if "text" in document:
                words = document["text"].split()
                word_count = len(words)
                word_count_to_doc_count[word_count] += 1
                total_word_count += word_count

    # Prepare the output dictionary
    output_data = {"word_counts": dict(word_count_to_doc_count), "total_num_words": total_word_count}

    # Write the result to a JSON file
    with output_file_path.open("w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=4)
