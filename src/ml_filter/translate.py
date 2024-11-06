import re
from abc import ABC, abstractmethod
from pathlib import Path

import deepl
import openai
import yaml

from constants import EUROPEAN_LANGUAGES


class Translator(ABC):
    """An abstract class for translating text into multiple languages."""

    def __init__(
        self,
        api_key: str,
        input_path: Path,
        source_language_code: str,
        target_language_codes: list[str],
        ignore_tag: str | None,
    ):
        """Initialize the translation class with the given parameters.

        Args:
            api_key (str): The API key for the translation service.
            input_path (Path): The path to the input file.
            source_language (str): The source language code.
            languages (list[str]): A list of target language codes.
            tag_to_ignore (str | None): A tag to ignore during translation, if any.
        """
        self.api_key = api_key
        self.input_path = input_path
        self.source_language_code = source_language_code
        self.language_codes = language_codes
        self.ignore_tag = ignore_tag

    @abstractmethod
    def translate(self) -> dict[str, str]:
        """Abstract method that performs translation and returns results."""
        raise not ImplementedError

    @property
    @abstractmethod
    def client(self):
        """An abstract property for the client."""
        raise NotImplementedError

    def write_output(self, output_path: Path, data: dict[str, str]) -> None:
        """Writes the provided data to YAML files in the specified output directory.
        This method ensures that the output directory exists and then writes each
        entry in the data dictionary to a separate YAML file. The filename is
        constructed using the key from the dictionary.

        Args:
            output_path (Path): The path to the directory where the output files
                    will be written.
            data (dict[str, str]): A dictionary where each key is a language code
                       and each value is the corresponding text to be
                       written to the file.
        Returns:
            None
        """

        output_path.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists
        for lang, text in data.items():
            file_path = output_path / f"educational_prompt_{lang}.yaml"
            with file_path.open("w", encoding="utf-8") as file:
                yaml.dump({"prompt": text}, file, default_flow_style=False, allow_unicode=True)

    def _load_data(self) -> dict:
        with open(self.input_path, "r") as file:
            data = yaml.safe_load(file)
        return data


class DeepLTranslator(Translator):
    """A class to translate text into multiple languages using the DeepL API."""

    def __init__(
        self,
        api_key: str,
        input_path: Path,
        source_language_code: str,
        language_codes: list[str],
        tag_to_ignore: str | None,
    ):
        super().__init__(api_key, input_path, source_language_code, language_codes, tag_to_ignore)
        self._client = deepl.Translator(self.api_key)

    @property
    def client(self):
        return self._client

    def translate(self) -> dict[str, str]:
        """Translate the given text into multiple languages using the DeepL API."""
        translated_data = {}
        data = self._load_data()
        text = data["prompt"]

        ignore_tag = self._get_ignore_tag(ignore_tag=self.ignore_tag)
        tag_handling = self._get_tag_handling_strategy(ignore_tag=self.ignore_tag)

        if ignore_tag is not None:
            ignore_tag = [ignore_tag]

        for lang_code in self.language_codes:
            result = self.client.translate_text(
                text,
                source_lang=self.source_language_code,
                target_lang=target_lang_code,
                tag_handling=tag_handling,
                ignore_tags=ignore_tag,
            )
            translated_data[lang_code] = result.text

        return translated_data

    def _get_tag_handling_strategy(self, ignore_tag: str | None):
        return "xml" if ignore_tag is not None else None

    def _get_ignore_tag(self, ignore_tag: str | None) -> str | None:
        if ignore_tag is None:
            return None

        match = re.search(r"<(.*)>", ignore_tag)

        if match:
            result = match.group(1)
            ignore_tag = result
        else:
            raise ValueError("No tag found in the text.")

        return ignore_tag


class OpenAITranslator(Translator):
    """A class to translate text into multiple languages using the OpenAI API."""

    def __init__(
        self,
        api_key: str,
        input_path: Path,
        source_language: str,
        target_languages: list[str],
        tag_to_ignore: str | None,
    ):
        super().__init__(api_key, input_path, source_language, languages, tag_to_ignore)

        self._client = openai.OpenAI(api_key=api_key)

    @property
    def client(self):
        return self._client

    def translate(self) -> dict[str, str]:
        """Translate the given text into multiple languages using the OpenAI API."""
        translated_data = {}
        data = self._load_data()
        text = data["prompt"]

        ignore_text = self._get_ignore_text(tag_to_ignore=self.ignore_tag)

        for lang_code in self.language_codes:
            language = EUROPEAN_LANGUAGES[lang_code]
            prompt = f"""Translate the following text into {language}{ignore_text}: {text}."""
            # Call the API
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a translation assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            translated_data[lang_code] = response.choices[0].message

        return translated_data

    def _get_ignore_text(self, tag_to_ignore: str | None) -> str:
        if tag_to_ignore is not None:
            closing_tag_to_ignore = re.sub(r"<(\w+)>", r"</\1>", tag_to_ignore)
            # Add whitespace at the beginning since the text is appended to the prompt
            ignore_text = f". Text that is within '{self.ignore_tag} {closing_tag_to_ignore}' should not be translated"
        else:
            ignore_text = ""
        return ignore_text
