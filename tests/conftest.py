from pathlib import Path

import pytest

from ml_filter.translate import DeepLClient, OpenAIClient, Translator


@pytest.fixture
def deepl_translator():
    deepl_client = DeepLClient(api_key="fake_key", ignore_tag_text="notranslate")
    return Translator(
        client=deepl_client,
        input_path=Path("tests/resources/data/translate_en.yaml"),
    )


@pytest.fixture
def openai_translator():
    openai_client = OpenAIClient(api_key="fake_key", ignore_tag_text="notranslate")
    return Translator(
        client=openai_client,
        input_path=Path("tests/resources/data/translate_en.yaml"),
    )
