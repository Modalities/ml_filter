from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from ml_filter.utils.manipulate_prompt import add_target_langauge_to_prompt

# Mock constants
TARGET_LANGAUGE_PLACEHOLDER = "{##TARGET_LANGUAGE##}"
EUROPEAN_LANGUAGES = {
    "en": "English",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
}


@pytest.fixture
def create_input_yaml():
    """Creates a sample YAML file for testing."""
    data = {"prompt": "Translate the text to {##TARGET_LANGUAGE##}."}
    with TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input.yaml"
        with open(input_file, "w") as file:
            yaml.safe_dump(data, file)
        yield input_file


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
