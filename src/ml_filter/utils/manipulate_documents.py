import json
import os
from pathlib import Path

import yaml

from constants import EUROPEAN_LANGUAGES, TARGET_LANGAUGE_PLACEHOLDER


def verify_jsonl_file_name_consistency(
    directory: Path,
    file_name_delimiter: str,
    file_name_keep_idx: list[int],
) -> list[Path]:
    """Verifies the presence and naming consistency of JSONL files in a directory.
    It checks that specific components of the file names (as determined by indices in
    `file_name_keep_idx`) are consistent across all JSONL files.

    Args:
        directory (Path): The path to the directory to be checked.
        file_name_delimiter (str): The delimiter used to split the file names.
        file_name_keep_idx (list[int]): Indices of the components in the split file names to check.

    Returns:
        list[Path]: A list of JSONL file paths in the directory.

    Raises:
        ValueError: If the specified components of the file names do not match for all files.
    """
    # Get all JSONL files in the directory
    jsonl_files = [file for file in directory.iterdir() if file.suffix == ".jsonl"]
    if not jsonl_files:
        raise ValueError("No JSONL files found in the directory.")

    # Extract and compare the specified components from file names
    file_names = set()
    for f in jsonl_files:
        file_name = _costruct_file_name(
            file_path=f,
            file_name_delimiter=file_name_delimiter,
            file_name_keep_idx=file_name_keep_idx,
        )
        file_names.add(file_name)

    if len(file_names) != 1:
        raise ValueError(
            "The specified components of the file names do not match for all files. "
            f"Inconsistent components: {file_names}"
        )

    return jsonl_files


def merge_and_sort_jsonl_files(
    directory: Path,
    file_name_delimiter: str,
    file_name_keep_idx: list[int],
    document_key: str,
) -> None:
    """Merges and sorts JSONL files in a directory by the 'id' field.
    This function reads all JSONL files in the specified directory, merges their contents,
    sorts the documents by the 'id' field, and writes the sorted documents to a new JSONL file.
    The output file name is generated based on the first input file's name, keeping a specified
    number of entries from the end of the filename.
    Args:
        directory (Path): The directory containing the JSONL files to be merged and sorted.
        split_filename_by (str): The delimiter used to split the filename for generating the output filename.
        num_filename_entries_to_keep (int): The number of entries from the end of the filename
        to keep for the output filename.
        document_key (str): The key to sort the documents by.
    Raises:
        ValueError: If the number of filename entries to keep is greater than the number of filename entries
        in the first file's name.
    """

    jsonl_files = verify_jsonl_file_name_consistency(
        directory=directory,
        file_name_delimiter=file_name_delimiter,
        file_name_keep_idx=file_name_keep_idx,
    )
    documents = []
    file_name = _costruct_file_name(
        file_path=jsonl_files[0],
        file_name_delimiter=file_name_delimiter,
        file_name_keep_idx=file_name_keep_idx,
    )
    output_file = directory / f"merged_{file_name}.jsonl"
    # Read and collect documents from all files
    for file in jsonl_files:
        with open(os.path.join(directory, file), "r") as f:
            for line in f:
                doc = json.loads(line)
                documents.append(doc)

    # Sort documents by the 'id' field
    # Convert 'id' to an integer for sorting - It is assunmed that id is a integer
    # stored as string in the document
    sorted_documents = sorted(documents, key=lambda x: int(x[document_key]))

    # Write the sorted documents to the output JSONL file
    with open(output_file, "w") as f:
        for doc in sorted_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def _costruct_file_name(
    file_path: Path,
    file_name_keep_idx: list[int],
    file_name_delimiter: str,
) -> str:
    """Constructs a file name based on the specified components of the file name."""
    file_name_splits = file_path.stem.split(file_name_delimiter)
    try:
        file_name = file_name_delimiter.join([file_name_splits[i] for i in file_name_keep_idx])
    except IndexError:
        raise ValueError(
            f"File name '{file_path.stem}' does not have enough components to extract indices {file_name_keep_idx}."
        )
    return file_name


def add_target_language_to_prompt(input_file_path: Path, output_dir: Path) -> None:
    """Reads a YAML file, replaces '{##TARGET_LANGUAGE##}' in the 'prompt' key with a given value,
    and writes the result to a new file.
    Args:
        input_file_path (Path): Path to the input YAML file.
        output_dir (Path): Directory to save the updated YAML file.
    Raises:
        KeyError: If the 'prompt' key does not exist in the YAML file.
        Exception: If any other error occurs during file processing.
    """
    for language_code, language in EUROPEAN_LANGUAGES.items():
        try:
            # Read the YAML file
            with open(input_file_path, "r") as file:
                data = yaml.safe_load(file)

            # Check if 'prompt' key exists
            if "prompt" in data:
                original_prompt = data["prompt"]
                updated_prompt = original_prompt.replace(TARGET_LANGAUGE_PLACEHOLDER, language)
                data["prompt"] = updated_prompt

                # Save the updated YAML
                file_name = input_file_path.stem
                output_file_path = output_dir / f"{file_name }_{language_code}.yaml"
                with open(output_file_path, "w") as file:
                    yaml.safe_dump(data, file, default_flow_style=False)

                print(f"Updated 'prompt' saved to {output_dir}")
            else:
                raise KeyError("The 'prompt' key does not exist in the YAML file.")
        except Exception as e:
            print(f"The following error occurred: {e}.")
