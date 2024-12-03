import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from pydantic import FilePath

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
    def translate_text(self, text: str, source_language_code: str, target_language_code: str) -> dict[str, str]:
        """Translate the text and return results as a dictionary."""
        raise NotImplementedError

    def assert_source_language_available(self, source_language_code: str) -> None:
        """Checks if the source language is available in the predefined supported source language set.
        Raises a ValueError if the source language is not available.

        Args:
            source_language (str): Source language to validate.

        Raises:
            ValueError: If the source language is not available in the supported source language set.
        """
        if source_language_code not in self.supported_source_languages:
            raise ValueError(f"The source language {source_language_code} is not available.")

    def assert_target_language_available(self, target_language_code: list[str]) -> None:
        """Checks if the target language is available in the predefined supported target language set.
        Raises a ValueError if the language in the target_languages list is not available.

        Args:
            target_language_code (str): The target languages to validate.

        Raises:
            ValueError: If the target language is not available in the supported target language set.
        """
        if target_language_code not in self.supported_target_languages:
            raise ValueError(f"The target language is not available: {target_language_code}.")


class Translator:
    """A class for translating text into multiple languages using a specified client."""

    def __init__(self, client: TranslationClient):
        """Initialize the translation class with the client and parameters."""
        self.client = client

    def translate_text(self, text: str, source_language_code: str, target_language_code: str) -> str:
        """Translate the text into the target language using the specified client."""
        return self.client.translate_text(text, source_language_code, target_language_code)

    def translate_jsonl_to_multiple_languages(
        self,
        input_file_path: FilePath,
        output_folder_path: Path,
        source_language_code: str,
        target_language_codes: list[str],
    ) -> None:
        """
        Translates the 'text' field in each JSON document from a JSONL input file into
        multiple target languages using a generator for data loading. Creates one JSONL file
        per target language, processing the input file in a single pass for efficiency.

        Args:
            input_file_path (FilePath): Path to the JSONL input file.
            output_folder_path (Path): Path to the folder for output files.
            source_language_code (str): The source language code.
            target_language_codes (list[str]): List of target language codes.
        """
        # Ensure output folder exists
        output_folder_path.mkdir(parents=True, exist_ok=True)

        def read_jsonl(file_path):
            """Generator to yield JSON objects from a JSONL file."""
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    yield json.loads(line)

        # Open output files for all target languages
        output_files = {
            language_code: open(
                output_folder_path / f"{input_file_path.stem}_{language_code}.jsonl", "w", encoding="utf-8"
            )
            for language_code in target_language_codes
        }

        try:
            # Use the generator to read and process the input JSONL file
            for document in read_jsonl(input_file_path):
                text = document.get("text")
                if not isinstance(text, str):
                    raise ValueError("Each document must have a 'text' field with a string value.")

                # Translate the text to all target languages
                translated_documents = {
                    lang: {**document, "text": self.translate_text(text, source_language_code, lang)}
                    for lang in target_language_codes
                }

                # Write the translated documents to their respective files
                for lang, translated_doc in translated_documents.items():
                    json.dump(translated_doc, output_files[lang], ensure_ascii=False)
                    output_files[lang].write("\n")
        finally:
            # Ensure all output files are properly closed
            for file in output_files.values():
                file.close()

    def translate_flat_yaml_to_multiple_languages(
        self,
        input_file_path: FilePath,
        output_folder_path: Path,
        source_language_code: str,
        target_language_codes: list[str],
    ) -> None:
        """Translates the text (i.e., the value fields) in the input file into multiple
        languages using the specified client. We create one file per target language, where
        the file contains the translated text. The file name is constructed as follows:
        <original_file_name>_{language_code}.yaml.
        Raises an error if the value fields are not strings."""
        data = self._load_yaml_data(input_file_path)
        translated_data = {}
        for target_language_code in target_language_codes:
            translated_data[target_language_code] = {}
            for key, value in data.items():
                if not isinstance(value, str):
                    raise ValueError(f"Value for key '{key}' is not a string.")
                translated_data[target_language_code][key] = self.translate_text(
                    value, source_language_code, target_language_code
                )

        for language_code, data in translated_data.items():
            output_file_path = output_folder_path / f"{input_file_path.stem}_{language_code}.yaml"
            self._write_output(output_file_path, data)

    @staticmethod
    def _load_yaml_data(input_path: Path) -> dict:
        """Loads data from a YAML file specified by the input path.

        Returns:
            dict: The data loaded from the YAML file.
        """
        with open(input_path, "r") as file:
            data = yaml.safe_load(file)
        return data

    @staticmethod
    def _write_output(output_file_path: FilePath, data: dict[str, str]):
        """Writes translated data to YAML files in the specified output directory."""
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        with output_file_path.open("w", encoding="utf-8") as file:
            yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


class DeepLClient(TranslationClient):
    """Client for the DeepL API."""

    def __init__(self, api_key: str, ignore_tag_text: str | None = None):
        """Initializes the DeepL client with the API key and optional ignore tag text.

        Args:
            api_key (str): The API key for the DeepL client.
            ignore_tag_text (str | None, optional): The XML tag to ignore.
            Note that the tag must be specified with the <>. Defaults to None.
        """
        super().__init__(api_key=api_key, ignore_tag_text=ignore_tag_text)
        import deepl

        self.client = deepl.Translator(api_key)

    @property
    def supported_source_languages(self) -> list[str]:
        """Retrieves a list of supported source languages.

        Returns:
            list[str]: A list of language codes representing the supported source languages.
        """
        return [lang.code.lower() for lang in self.client.get_source_languages()]

    @property
    def supported_target_languages(self) -> list[str]:
        """
        Retrieve a list of supported target languages.

        This method queries the client to get the available target languages and returns their codes.

        Returns:
            list[str]: A list of language codes representing the supported target languages.
        """
        return [lang.code.lower() for lang in self.client.get_target_languages()]

    def translate_text(self, text: str, source_language_code: str, target_language_code: str) -> str:
        """Translates the given text from the source language to the specified target language using the DeepL client.

        Args:
            text (str): The text to be translated.
            source_language_code (str): The language code of the source text.
            target_language_code (str): The language code to which the text should be translated.

        Returns:
            str: The translated text.

        Raises:
            ValueError: If the source language or the target language  is not available.
        """
        self.assert_source_language_available(source_language_code=source_language_code)
        self.assert_target_language_available(target_language_code=target_language_code)
        tag_handling = "xml" if self.ignore_tag_text is not None else None

        ignore_tags = None if self.ignore_tag_text is None else [self.ignore_tag_text]
        result = self.client.translate_text(
            text,
            source_lang=source_language_code,
            target_lang=target_language_code,
            tag_handling=tag_handling,
            ignore_tags=ignore_tags,
        )
        return result.text


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

    def translate_text(self, text: str, source_language_code: str, target_language_code: str) -> str:
        """Translates the given text from the source language to multiple target languages using the OpenAI client.

        Args:
            text (str): The text to be translated.
            source_language_code (str): The language code of the source text.
            target_language_code (str): The language code to which the text should be translated.

        Returns:
            str: The translated text.

        Raises:
            ValueError: If the source language or any of the target languages are not available.
        """

        self.assert_source_language_available(source_language_code=source_language_code)
        self.assert_target_language_available(target_language_code=target_language_code)
        ignore_text = self._get_ignore_text()

        language = EUROPEAN_LANGUAGES[target_language_code]
        prompt = f"Translate the following text into {language}.{ignore_text} The text: {text}"
        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a translation assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        translated_text = response.choices[0].message["content"]
        return translated_text

    def _get_ignore_text(self) -> str:
        """Helper to create ignore text if an ignore tag is specified."""
        if self.ignore_tag_text:
            opening_tag = f"<{self.ignore_tag_text}>"
            closing_tag = f"</{self.ignore_tag_text}>"
            return f" Text within '{opening_tag} {closing_tag}' should not be translated."
        return ""


class TranslatorFactory:
    @staticmethod
    def get_openai_translator(ignore_tag_text: str | None = None) -> Translator:
        api_key = TranslatorFactory._get_api_key("OPENAI_API_KEY")
        client = OpenAIClient(api_key, ignore_tag_text)
        return Translator(client)

    @staticmethod
    def get_deepl_translator(ignore_tag_text: str | None = None) -> Translator:
        api_key = TranslatorFactory._get_api_key("DEEPL_API_KEY")
        client = DeepLClient(api_key, ignore_tag_text)
        return Translator(client)

    def _get_api_key(evn_variable_name: str):
        api_key = os.getenv(evn_variable_name)
        if api_key is None or api_key == "":
            raise EnvironmentError(
                f"API key in environment variable '{evn_variable_name}' is not set. Please set it to continue."
            )
        return api_key
