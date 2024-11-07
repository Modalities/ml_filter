from abc import ABC, abstractmethod
from pathlib import Path

import yaml

from constants import EUROPEAN_LANGUAGES


class TranslationClient(ABC):
    """An interface for translation clients."""

    def __init__(self, api_key: str, ignore_tag_text: str | None = None):
        self.api_key = api_key
        self.ignore_tag_text = ignore_tag_text

    @property
    @abstractmethod
    def supported_source_languages(self) -> list[str]:
        """A property that should return a list of supported languages."""
        raise NotImplementedError

    @property
    @abstractmethod
    def supported_target_languages(self) -> list[str]:
        """A property that should return a list of supported languages."""
        raise NotImplementedError

    @abstractmethod
    def translate_text(self, text: str, source_language_code: str, target_language_codes: list[str]) -> dict[str, str]:
        """Translate the text and return results as a dictionary."""
        raise NotImplementedError

    def assert_source_language_available(self, source_language_code: str) -> None:
        """Checks if the source language is available in the predefined EUROPEAN_LANGUAGES set.
        Raises a ValueError if the source language is not available.

        Args:
            source_language (str): Source language to validate.

        Raises:
            ValueError: If the source language is not available in EUROPEAN_LANGUAGES.
        """
        if source_language_code not in self.supported_source_languages:
            raise ValueError(f"The source language {source_language_code} is not available.")

    def assert_target_languages_are_available(self, target_language_codes: list[str]) -> None:
        """Checks if all target languages are available in the predefined EUROPEAN_LANGUAGES set.
        Raises a ValueError if any language in the target_languages list is not available.

        Args:
            super: The superclass or any required context for the method.
            target_languages (list[str]): List of target languages to validate.

        Raises:
            ValueError: If any of the target languages is not available in EUROPEAN_LANGUAGES.
        """
        unavailable_languages = [
            lang_code for lang_code in target_language_codes if lang_code not in self.supported_target_languages
        ]

        if unavailable_languages:
            raise ValueError(f"The following target languages are not available: {', '.join(unavailable_languages)}")


class Translator:
    """A class for translating text into multiple languages using a specified client."""

    def __init__(self, client: TranslationClient, input_path: Path):
        """Initialize the translation class with the client and parameters."""
        self.client = client
        self.input_path = input_path

    def translate_text(self, text: str, source_language_code: str, target_language_codes: list[str]) -> dict[str, str]:
        """Translate the text into multiple languages using the specified client."""
        return self.client.translate_text(text, source_language_code, target_language_codes)

    def write_output(self, output_path: Path, data: dict[str, str]) -> None:
        """Writes translated data to YAML files in the specified output directory."""
        output_path.mkdir(parents=True, exist_ok=True)
        for lang, text in data.items():
            file_path = output_path / f"educational_prompt_{lang}.yaml"
            with file_path.open("w", encoding="utf-8") as file:
                yaml.dump({"prompt": text}, file, default_flow_style=False, allow_unicode=True)

    def load_data(self) -> dict:
        """Loads data from a YAML file specified by the input path.

        Returns:
            dict: The data loaded from the YAML file.
        """
        with open(self.input_path, "r") as file:
            data = yaml.safe_load(file)
        return data


class DeepLClient(TranslationClient):
    """Client for the DeepL API."""

    def __init__(self, api_key: str, ignore_tag_text: str | None = None):
        super().__init__(api_key=api_key, ignore_tag_text=ignore_tag_text)
        import deepl

        self.client = deepl.Translator(api_key)

    @property
    def supported_source_languages(self) -> list[str]:
        """Retrieves a list of supported source languages.

        Returns:
            list[str]: A list of language codes representing the supported source languages.
        """
        return [lang.code for lang in self.client.get_source_languages()]

    @property
    def supported_target_languages(self) -> list[str]:
        """
        Retrieve a list of supported target languages.

        This method queries the client to get the available target languages and returns their codes.

        Returns:
            list[str]: A list of language codes representing the supported target languages.
        """
        return [lang.code for lang in self.client.get_target_languages()]

    def translate_text(self, text: str, source_language_code: str, target_language_codes: list[str]) -> dict[str, str]:
        """Translates the given text from the source language to multiple target languages using the DeepL client.

        Args:
            text (str): The text to be translated.
            source_language_code (str): The language code of the source text.
            target_language_codes (list[str]): A list of language codes to which the text should be translated.

        Returns:
            dict[str, str]: A dictionary where the keys are target language codes
              and the values are the translated texts.

        Raises:
            ValueError: If the source language or any of the target languages are not available.
        """
        self.assert_source_language_available(source_language_code=source_language_code)
        self.assert_target_languages_are_available(target_language_codes=target_language_codes)
        translated_data = {}
        ignore_tag = self._get_ignore_tag(self.ignore_tag_text)
        tag_handling = "xml" if ignore_tag else None

        for target_lang_code in target_language_codes:
            result = self.client.translate_text(
                text,
                source_lang=source_language_code,
                target_lang=target_lang_code,
                tag_handling=tag_handling,
                ignore_tags=[ignore_tag] if ignore_tag else None,
            )
            translated_data[target_lang_code] = result.text

        return translated_data

    def _get_ignore_tag(self, ignore_tag_text: str | None) -> str | None:
        """Create the ignore tag if ignore tag text provided."""
        if ignore_tag_text is None:
            return None
        return f"<{ignore_tag_text}>"


class OpenAIClient(TranslationClient):
    """Client for the OpenAI API."""

    def __init__(self, api_key: str, ignore_tag_text: str | None = None):
        super().__init__(api_key=api_key, ignore_tag_text=ignore_tag_text)
        import openai

        self.client = openai.OpenAI(api_key=api_key)

    @property
    def supported_source_languages(self) -> list[str]:
        """Retrieves a list of supported source languages.

        Returns:
            list[str]: A list of language codes representing the supported source languages.
        """
        return list(EUROPEAN_LANGUAGES.keys())

    @property
    def supported_target_languages(self) -> list[str]:
        """
        Retrieve a list of supported target languages.

        This method queries the client to get the available target languages and returns their codes.

        Returns:
            list[str]: A list of language codes representing the supported target languages.
        """
        return list(EUROPEAN_LANGUAGES.keys())

    def translate_text(self, text: str, source_language_code: str, target_language_codes: list[str]) -> dict[str, str]:
        """Translates the given text from the source language to multiple target languages using the OpenAI client.

        Args:
            text (str): The text to be translated.
            source_language_code (str): The language code of the source text.
            target_language_codes (list[str]): A list of language codes to which the text should be translated.

        Returns:
            dict[str, str]: A dictionary where the keys are target language codes
              and the values are the translated texts.

        Raises:
            ValueError: If the source language or any of the target languages are not available.
        """

        self.assert_source_language_available(source_language_code=source_language_code)
        self.assert_target_languages_are_available(target_language_codes=target_language_codes)
        translated_data = {}
        ignore_text = self._get_ignore_text()

        for target_lang in target_language_codes:
            language = EUROPEAN_LANGUAGES[target_lang]
            prompt = f"Translate the following text into {language}.{ignore_text} The text: {text}"
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a translation assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            translated_data[target_lang] = response.choices[0].message["content"]

        return translated_data

    def _get_ignore_text(self) -> str:
        """Helper to create ignore text if an ignore tag is specified."""
        if self.ignore_tag_text:
            opening_tag = f"<{self.ignore_tag_text}>"
            closing_tag = f"</{self.ignore_tag_text}>"
            return f" Text within '{opening_tag} {closing_tag}' should not be translated."
        return ""
