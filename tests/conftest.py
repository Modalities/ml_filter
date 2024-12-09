import json
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import yaml

from ml_filter.translate import DeepLClient, OpenAIClient, Translator


@pytest.fixture
def deepl_translator():
    deepl_client = DeepLClient(api_key="fake_key", ignore_tag_text="notranslate")
    return Translator(client=deepl_client)


@pytest.fixture
def openai_translator():
    openai_client = OpenAIClient(api_key="fake_key", ignore_tag_text="notranslate")
    return Translator(client=openai_client)


@pytest.fixture
def create_input_yaml():
    """Creates a sample YAML file for testing."""
    data = {"prompt": "Translate the text to {##TARGET_LANGUAGE##}."}
    with TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "input.yaml"
        with open(input_file, "w") as file:
            yaml.safe_dump(data, file)
        yield input_file


@pytest.fixture
def mock_translate_text():
    """Fixture to mock the translate_text method."""

    def translate_mock(text, source_language, target_language):
        return f"{text} translated to {target_language}"

    return translate_mock


@pytest.fixture
def temporary_jsonl_file(tmp_path):
    """Fixture to create a temporary JSONL input file."""
    file_path = tmp_path / "input.jsonl"
    documents = [
        {"text": "Hello, world!", "id": 1},
        {"text": "How are you?", "id": 2},
    ]
    with open(file_path, "w", encoding="utf-8") as f:
        for doc in documents:
            f.write(json.dumps(doc) + "\n")
    return file_path


@pytest.fixture
def tmp_input_file(tmp_path):
    """
    Creates a temporary input JSONL file for testing.
    """
    input_file = tmp_path / "test_input.jsonl"
    content = [
        {"text": "This is a test document."},
        {"text": "Another test."},
        {"text": "Yet another example of a document with more words."},
        {"text": "Short one."},
    ]
    with input_file.open("w", encoding="utf-8") as f:
        for entry in content:
            f.write(json.dumps(entry) + "\n")
    return input_file


@pytest.fixture
def output_folder(tmp_path):
    """Fixture to create a temporary output folder."""
    output_path = tmp_path / "output"
    output_path.mkdir()
    return output_path


@pytest.fixture
def tmp_output_file(tmp_path):
    """
    Provides a temporary output file path.
    """
    return tmp_path / "test_output.json"
