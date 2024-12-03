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
