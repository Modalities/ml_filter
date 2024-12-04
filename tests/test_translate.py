from dataclasses import dataclass
from unittest.mock import Mock

import deepl
import openai
import pytest
import yaml

from ml_filter.translate import Translator


@dataclass
class Message:
    content: str


@dataclass
class Choice:
    message: Message


@dataclass
class Langauge:
    code: str


@dataclass
class TextResult:
    text: str | None = None
    choices: list[Choice] | None = None


def _test_raises_exception(text, translator: Translator):
    for source_lang_code, target_lang_code in zip(["xx2", "de"], [["xx2", "fr"], ["xx2", "fr"]]):
        with pytest.raises(Exception):
            translator.translate_text(
                text=text,
                source_language_code=source_lang_code,
                target_language_code=target_lang_code,
            )


def test_deepl_translate(deepl_translator):
    """Test the translation of text into multiple languages."""

    with open("tests/resources/data/translate_en.yaml", "r") as file:
        data = yaml.safe_load(file)

    text = data["prompt"]
    source_lang = "en"
    target_lang = "de"

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
        target_language_code=target_lang,
    )
    assert translated_data == data["prompt"]

    _test_raises_exception(text=text, translator=deepl_translator)


def test_openai_translate(openai_translator):
    with open("tests/resources/data/translate_en.yaml", "r") as file:
        data = yaml.safe_load(file)

    text = data["prompt"]
    source_lang = "en"
    target_lang = "fr"
    openai_client = Mock(spec=openai.OpenAI)
    openai_client.chat = Mock()
    openai_client.chat.completions = Mock()
    openai_client.translate_text = lambda text, source_lang, target_lang, tag_handling, ignore_tags: TextResult(
        text=text
    )
    openai_client.chat.completions.create = lambda model, messages: TextResult(
        choices=[Choice(message=Message(messages[1]["content"]))]
    )
    openai_translator.client.client = openai_client

    translated_data = openai_translator.translate_text(
        text=text,
        source_language_code=source_lang,
        target_language_code=target_lang,
    )
    expected_data = (
        f"Translate the following text into French. "
        f"Text within '<notranslate> </notranslate>' should not be translated. "
        f"The text: {data['prompt']}"
    )

    assert translated_data == expected_data

    _test_raises_exception(text=text, translator=openai_translator)
