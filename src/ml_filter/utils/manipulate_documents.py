import json
import os
from pathlib import Path

import yaml

from constants import EUROPEAN_LANGUAGES, TARGET_LANGAUGE_PLACEHOLDER


def verify_jsonl_file_name_consistency(directory: Path) -> list[Path]:
    """Verifies the presence and naming consistency of JSONL files in a directory.
    It also verifies that the last two components of the file names (when split by underscores)
    are consistent across all JSONL files.
    Args:
        directory (Path): The path to the directory to be checked.
    Returns:
        list[Path]: A list of JSONL paths in the directory.
    Raises:
        ValueError: If the last two components of the file names do not match for all files.
    """
    jsonl_files = [file for file in directory.iterdir() if file.suffix == ".jsonl"]

    # Split file names and check the last two components
    components = [f.stem.split("_")[-2:] for f in jsonl_files]
    if len(set(map(tuple, components))) != 1:
        raise ValueError("The last two components of the file names do not match for all files.")

    return jsonl_files


def merge_and_sort_files(directory: Path, split_filename_by: str = "_", num_filename_entries_to_keep: int = 2) -> None:
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
    Raises:
        ValueError: If the number of filename entries to keep is greater than the number of filename entries
        in the first file's name.
    """

    jsonl_files = verify_files(directory)
    documents = []
    file_name = jsonl_files[0].stem.split(split_filename_by)
    if len(file_name) >= num_filename_entries_to_keep:
        file_name = file_name[-num_filename_entries_to_keep:]
    else:
        raise ValueError("The number of filename entries to keep is greater than the number of filename entries.")
    file_name = "_".join(file_name)
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
    sorted_documents = sorted(documents, key=lambda x: int(x["id"]))

    # Write the sorted documents to the output JSONL file
    with open(output_file, "w") as f:
        for doc in sorted_documents:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def add_target_langauge_to_prompt(input_file_path: Path, output_dir: Path) -> None:
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
