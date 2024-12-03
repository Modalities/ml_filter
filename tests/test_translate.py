import json
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


def test_translate_jsonl_to_multiple_languages(mock_translate_text, temporary_jsonl_file, output_folder):
    """Test the translate_jsonl_to_multiple_languages method."""

    class MockTranslationClient:
        def translate_text(self, text, source_language, target_language):
            return mock_translate_text(text, source_language, target_language)

        def translate_jsonl_to_multiple_languages(
            self,
            input_file_path,
            output_folder_path,
            source_language_code,
            target_language_codes,
        ):
            translator = Translator(client=self)
            translator.translate_jsonl_to_multiple_languages(
                input_file_path=input_file_path,
                output_folder_path=output_folder_path,
                source_language_code=source_language_code,
                target_language_codes=target_language_codes,
            )

        @property
        def supported_source_languages(self):
            return ["en"]

        @property
        def supported_target_languages(self):
            return ["fr", "es"]

    mock_client = MockTranslationClient()
    translator = Translator(client=mock_client)

    source_language = "en"
    target_languages = ["fr", "es"]

    # Call the method under test
    translator.translate_jsonl_to_multiple_languages(
        input_file_path=temporary_jsonl_file,
        output_folder_path=output_folder,
        source_language_code=source_language,
        target_language_codes=target_languages,
    )

    # Verify output files
    for lang in target_languages:
        output_file = output_folder / f"input_{lang}.jsonl"
        assert output_file.exists()

        with open(output_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) == 2  # Ensure two documents are translated

            for i, line in enumerate(lines):
                doc = json.loads(line)
                expected_text = (
                    f"Hello, world! translated to {lang}" if i == 0 else f"How are you? translated to {lang}"
                )
                assert doc["text"] == expected_text

                # Check if other fields are preserved
                assert "id" in doc
                assert doc["id"] == i + 1


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
