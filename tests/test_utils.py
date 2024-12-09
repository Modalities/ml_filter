import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from ml_filter.utils.manipulate_documents import add_target_langauge_to_prompt, merge_and_sort_files, verify_files
from ml_filter.utils.statistics import compute_num_words_and_chars_in_jsonl

# Mock constants
TARGET_LANGAUGE_PLACEHOLDER = "{##TARGET_LANGUAGE##}"
EUROPEAN_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


def test_verify_files_consistent(tmp_jsonl_directory: tuple[Path, list[Path], list[Path]]):
    """Test that verify_files works for consistent file naming."""
    directory, consistent_files, _ = tmp_jsonl_directory

    # Remove the inconsistent file
    for f in directory.iterdir():
        if "different_suffix" in f.stem:
            f.unlink()

    result = verify_files(directory)
    assert len(result) == len(consistent_files)
    assert sorted(result) == sorted(consistent_files)


def test_verify_files_inconsistent(tmp_jsonl_directory):
    """Test that verify_files raises ValueError for inconsistent file naming."""
    directory, _, _ = tmp_jsonl_directory

    with pytest.raises(ValueError, match="The last two components of the file names do not match for all files."):
        verify_files(directory)


def test_verify_files_empty_directory(tmp_path):
    """Test that verify_files works with an empty directory."""
    with pytest.raises(ValueError, match="The last two components of the file names do not match for all files."):
        verify_files(tmp_path)


def test_merge_and_sort_files(merge_files_tmp_directory: Path):
    tmp_path = merge_files_tmp_directory

    merge_and_sort_files(tmp_path, split_filename_by="_", num_filename_entries_to_keep=2)

    # Verify output file
    expected_output_file = tmp_path / "merged_temp_file.jsonl"
    assert expected_output_file.exists(), "Output file not found!"

    # Read and verify the contents of the output file
    with open(expected_output_file, "r") as f:
        output_data = [json.loads(line) for line in f]

    # Expected sorted content
    expected_data = [
        {"id": "1", "value": "first"},
        {"id": "2", "value": "second"},
        {"id": "3", "value": "third"},
        {"id": "10", "value": "tenth"},
    ]

    assert output_data == expected_data, "The output data does not match the expected sorted data."


def test_add_target_language_to_prompt(create_input_yaml: Path):
    """Tests the add_target_langauge_to_prompt function."""
    input_file_path = create_input_yaml

    with TemporaryDirectory() as temp_output_dir:
        output_dir = Path(temp_output_dir)

        # Call the function
        add_target_langauge_to_prompt(input_file_path, output_dir)

        # Check generated files for all languages
        for lang_code, lang_name in EUROPEAN_LANGUAGES.items():
            output_file_path = output_dir / f"input_{lang_code}.yaml"
            assert output_file_path.exists(), f"Output file for {lang_name} not created."

            # Verify content
            with open(output_file_path, "r") as file:
                output_data = yaml.safe_load(file)
            expected_prompt = f"Translate the text to {lang_name}."
            assert output_data["prompt"] == expected_prompt, f"Prompt content for {lang_name} is incorrect."


def test_compute_num_words_and_chars_in_jsonl(tmp_input_file: Path, tmp_output_file: Path):
    # Call the function to compute the statistics
    compute_num_words_and_chars_in_jsonl(input_file_path=tmp_input_file, output_file_path=tmp_output_file)

    # Verify the output file exists
    assert tmp_output_file.exists(), "Output file was not created."

    # Load the output JSON data
    with tmp_output_file.open("r", encoding="utf-8") as f:
        output_data = json.load(f)

    # Expected word counts and total word count
    expected_word_counts = {
        5: 1,  # "This is a test document." (5 words)
        2: 2,  # "Another test.", "Short one." (2 words)
        9: 1,  # "Yet another example of a document with more words." (9 words)
    }
    expected_char_counts = {
        24: 1,  # "This is a test document." (27 characters)
        13: 1,  # "Another test." (12 characters)
        10: 1,  # "Short one." (10 characters)
        50: 1,  # "Yet another example of a document with more words." (48 characters)
    }
    expected_total_num_words = sum(key * value for key, value in expected_word_counts.items())
    expected_total_num_chars = sum([key for key in expected_char_counts.keys()])
    # output_data["total_num_words"] is saved as json, i.e., the keys are strings.
    # Therefore, we need to convert the keys to strings.
    expected_word_counts = {str(key): value for key, value in expected_word_counts.items()}

    # Validate word counts
    assert output_data["word_counts"] == dict(expected_word_counts), "Word counts  do not match expected values."

    # Validate total word count
    assert output_data["total_num_words"] == expected_total_num_words, "Total word count does not match expected value."

    # Validate total char count
    assert output_data["total_num_chars"] == expected_total_num_chars, "Total char count does not match expected value."
