from dataclasses import dataclass
from unittest.mock import Mock

from deepl import Translator as DeeepLTranslatorClient
from openai import OpenAI


@dataclass
class Choice:
    message: str


@dataclass
class TextResult:
    text: str | None = None
    choices: list[Choice] | None = None


def test_deep_translate(deepl_translator):
    """Test the translation of text into multiple languages."""
    data = deepl_translator._load_data()

    deepl_client = Mock(spec=DeeepLTranslatorClient)
    deepl_client.translate_text = lambda text, source_lang, target_lang, tag_handling, ignore_tags: TextResult(
        text=text
    )
    deepl_translator._client = deepl_client

    translated_data = deepl_translator.translate()
    assert translated_data == {
        "de": data["prompt"],
        "fr": data["prompt"],
    }


def test_openai_translate(openai_translator):
    data = openai_translator._load_data()
    openai_client = Mock(spec=OpenAI)
    openai_client.chat = Mock()
    openai_client.chat.completions = Mock()
    openai_client.chat.completions.create = lambda model, messages: TextResult(choices=[Choice(message=messages[1])])
    openai_translator._client = openai_client

    translated_data = openai_translator.translate()
    expected_data = {
        "de": {
            "role": "user",
            "content": f"""Translate the following text into German. Text that is within '<notranslate> </notranslate>'
            should not be translated: {data['prompt']}""",
        },
        "fr": {
            "role": "user",
            "content": f"""Translate the following text into French. Text that is within '<notranslate> </notranslate>'
            should not be translated: {data['prompt']}""",
        },
    }

    assert translated_data == expected_data


def test_deepl_get_ignore_tags(deepl_translator):
    ignore_tag = "<notranslate>"
    ignore_tag_text = deepl_translator._get_ignore_tag(ignore_tag=ignore_tag)

    assert ignore_tag_text == "notranslate"

    ignore_tag = "</notranslate>"
    ignore_tag_text = deepl_translator._get_ignore_tag(ignore_tag=ignore_tag)

    assert not ignore_tag_text == "notranslate"

    ignore_tag = "<notranslate> <other>"
    ignore_tag_text = deepl_translator._get_ignore_tag(ignore_tag=ignore_tag)

    assert not ignore_tag_text == "notranslate"


def test_deepl_get_tag_handling_strategy(deepl_translator):
    tag_handling_strategy = deepl_translator._get_tag_handling_strategy(ignore_tag=None)
    assert tag_handling_strategy is None

    tag_handling_strategy = deepl_translator._get_tag_handling_strategy(ignore_tag="<notranslate>")
    assert tag_handling_strategy == "xml"
