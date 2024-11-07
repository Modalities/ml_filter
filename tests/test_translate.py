from dataclasses import dataclass
from unittest.mock import Mock

import deepl
import openai
import pytest

from ml_filter.translate import DeepLClient, Translator


@dataclass
class Choice:
    message: str


@dataclass
class Langauge:
    code: str


@dataclass
class TextResult:
    text: str | None = None
    choices: list[Choice] | None = None


def _test_raises_exception(text, translator: Translator):
    for source_lang_code, target_lang_codes in zip(["xx2", "de"], [["xx2", "fr"], ["xx2", "fr"]]):
        with pytest.raises(Exception):
            translator.translate_text(
                text=text,
                source_language_code=source_lang_code,
                target_language_codes=target_lang_codes,
            )


def test_deep_translate(deepl_translator):
    """Test the translation of text into multiple languages."""
    data = deepl_translator.load_data()
    text = data["prompt"]
    source_lang = "en"
    target_langs = ["de", "fr"]

    deepl_client = Mock(spec=deepl.Translator)
    deepl_client.translate_text = lambda text, source_lang, target_lang, tag_handling, ignore_tags: TextResult(
        text=text
    )
    deepl_client.get_source_languages = lambda: [Langauge("en"), Langauge("de"), Langauge("fr")]
    deepl_client.get_target_languages = lambda: [Langauge("en"), Langauge("de"), Langauge("fr")]

    deepl_translator.client.client = deepl_client

    translated_data = deepl_translator.translate_text(
        text=text,
        source_language_code=source_lang,
        target_language_codes=target_langs,
    )
    assert translated_data == {
        "de": data["prompt"],
        "fr": data["prompt"],
    }

    _test_raises_exception(text=text, translator=deepl_translator)


def test_openai_translate(openai_translator):
    data = openai_translator.load_data()
    text = data["prompt"]
    source_lang = "en"
    target_langs = ["de", "fr"]
    openai_client = Mock(spec=openai.OpenAI)
    openai_client.chat = Mock()
    openai_client.chat.completions = Mock()
    openai_client.translate_text = lambda text, source_lang, target_lang, tag_handling, ignore_tags: TextResult(
        text=text
    )
    openai_client.chat.completions.create = lambda model, messages: TextResult(choices=[Choice(message=messages[1])])
    openai_translator.client.client = openai_client

    translated_data = openai_translator.translate_text(
        text=text,
        source_language_code=source_lang,
        target_language_codes=target_langs,
    )
    expected_data = {
        "de": (
            f"Translate the following text into German. "
            f"Text within '<notranslate> </notranslate>' should not be translated. "
            f"The text: {data['prompt']}"
        ),
        "fr": (
            f"Translate the following text into French. "
            f"Text within '<notranslate> </notranslate>' should not be translated. "
            f"The text: {data['prompt']}"
        ),
    }

    assert translated_data == expected_data

    _test_raises_exception(text=text, translator=openai_translator)


def test_deepl_get_ignore_tags():
    deepl_client = DeepLClient(api_key="fake_key", ignore_tag_text="notranslate")
    ignore_tag_text = "notranslate"
    ignore_tag = deepl_client._get_ignore_tag(ignore_tag_text=ignore_tag_text)

    assert ignore_tag == "<notranslate>"
