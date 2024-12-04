import json
from pathlib import Path
from tempfile import TemporaryDirectory

import yaml

from ml_filter.utils.manipulate_prompt import add_target_langauge_to_prompt
from ml_filter.utils.statistics import compute_num_words_in_jsonl

# Mock constants
TARGET_LANGAUGE_PLACEHOLDER = "{##TARGET_LANGUAGE##}"
EUROPEAN_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


def test_add_target_language_to_prompt(create_input_yaml):
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


def test_compute_num_words_in_jsonl(tmp_input_file, tmp_output_file):
    # Call the function to compute the statistics
    compute_num_words_in_jsonl(input_file_path=tmp_input_file, output_file_path=tmp_output_file)

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
    expected_total_num_words = sum(key * value for key, value in expected_word_counts.items())
    # output_data["total_num_words"] is saved as json, i.e., the keys are strings.
    # Therefore, we need to convert the keys to strings.
    expected_word_counts = {str(key): value for key, value in expected_word_counts.items()}

    # Validate word counts
    assert output_data["word_counts"] == dict(expected_word_counts), "Word counts  do not match expected values."

    # Validate total word count
    assert output_data["total_num_words"] == expected_total_num_words, "Total word count does not match expected value."
