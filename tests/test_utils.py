import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from ml_filter.utils.manipulate_documents import (
    add_target_language_to_prompt,
    merge_and_sort_jsonl_files,
    verify_jsonl_file_name_consistency,
)
from ml_filter.utils.statistics import (
    _count_words_in_file,
    _find_jsonl_files,
    compute_num_words_and_chars_in_jsonl,
    start_word_count_jsonl_files,
)

# Mock constants
TARGET_LANGAUGE_PLACEHOLDER = "{##TARGET_LANGUAGE##}"
EUROPEAN_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


def test_verify_files_consistent(
    tmp_jsonl_directory_with_consistent_and_inconsisten_files: tuple[
        Path, list[Path], list[Path], list[str], list[int]
    ],
):
    """Test that verify_files works for consistent file naming."""
    directory, consistent_files, _, _, file_name_keep_idx = tmp_jsonl_directory_with_consistent_and_inconsisten_files
    delimiter = "_"

    # Remove the inconsistent file
    for f in directory.iterdir():
        if "different_suffix" in f.stem:
            f.unlink()

    result = verify_jsonl_file_name_consistency(
        directory=directory,
        file_name_delimiter=delimiter,
        file_name_keep_idx=file_name_keep_idx,
    )
    assert len(result) == len(consistent_files)
    assert sorted(result) == sorted(consistent_files)


def test_verify_files_inconsistent(
    tmp_jsonl_directory_with_consistent_and_inconsisten_files: tuple[
        Path, list[Path], list[Path], list[str], list[int]
    ],
):
    """Test that verify_files raises ValueError for inconsistent file naming."""
    (
        directory,
        _,
        _,
        unique_file_name_stems,
        file_name_keep_idx,
    ) = tmp_jsonl_directory_with_consistent_and_inconsisten_files

    with pytest.raises(
        ValueError,
        match="The specified components of the file names do not match for all files. "
        f"Inconsistent components: {unique_file_name_stems}",
    ):
        verify_jsonl_file_name_consistency(
            directory=directory, file_name_delimiter="_", file_name_keep_idx=file_name_keep_idx
        )


def test_verify_files_empty_directory(tmp_path):
    """Test that verify_files works with an empty directory."""
    with pytest.raises(ValueError, match="No JSONL files found in the directory."):
        verify_jsonl_file_name_consistency(directory=tmp_path, file_name_delimiter="_", file_name_keep_idx=[])


def test_merge_and_sort_jsonl_files(merge_files_tmp_directory: tuple[Path, list[int]]):
    tmp_path, file_name_keep_idx = merge_files_tmp_directory

    merge_and_sort_jsonl_files(
        tmp_path, file_name_delimiter="_", file_name_keep_idx=file_name_keep_idx, document_key="id"
    )

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
        add_target_language_to_prompt(input_file_path=input_file_path, output_dir=output_dir)

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


def test_count_words_in_file(temporary_jsonl_file: Path):
    """Test the count_words_in_file function."""
    _, word_count = _count_words_in_file(temporary_jsonl_file)
    assert word_count == 5, 'Word count should be 5 "Hello world! "How are you?".'


def test_find_jsonl_files(tmp_jsonl_directory: Path):
    """Test the find_jsonl_files function."""
    files = _find_jsonl_files(directory_path=tmp_jsonl_directory)
    assert len(files) == 3, "There should be 3 JSONL files in the directory."
    for file in files:
        assert file.suffix == ".jsonl", "File extension should be .jsonl."


def test_start_word_count_jsonl_files(tmp_jsonl_directory: Path):
    """Test the process_files function."""
    output_file = tmp_jsonl_directory / "output.jsonl"
    start_word_count_jsonl_files(tmp_jsonl_directory, output_file)

    # Verify output file
    assert output_file.exists(), "Output file should exist."
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 3, "Output file should have 3 lines."
        for line in lines:
            data = json.loads(line)
            assert len(data) == 1, "Each line should contain one single key-value pair."


def test_process_nested_files(tmp_nested_jsonl_directory: tuple[Path, dict[str, int]], tmp_path: Path):
    """Test the process_files function with a nested directory structure."""
    root_directory, expected_word_counts = tmp_nested_jsonl_directory
    output_file = tmp_path / "nested_output.jsonl"
    start_word_count_jsonl_files(root_directory, output_file)

    # Verify output file
    assert output_file.exists(), "Output file should exist."
    with open(output_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) == 4, "Output file should have 4 lines (2 from root, 2 from sub_folder)."

        for line in lines:
            data = json.loads(line)
            for path, word_count in data.items():
                assert word_count == expected_word_counts[path], f"Word count for {Path(path).name} is incorrect."
