import pytest

from ml_filter.translate import DeepLTranslator, OpenAITranslator


@pytest.fixture
def deepl_translator():
    return DeepLTranslator(
        api_key="fake_key",
        language_codes=["de", "fr"],
        source_language_code="en",
        tag_to_ignore="<notranslate>",
        input_path="tests/resources/data/translate_en.yaml",
    )


@pytest.fixture
def openai_translator():
    return OpenAITranslator(
        api_key="fake_key",
        languages=["de", "fr"],
        source_language="en",
        tag_to_ignore="<notranslate>",
        input_path="tests/resources/data/translate_en.yaml",
    )
