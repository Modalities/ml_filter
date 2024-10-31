import re
from abc import ABC, abstractmethod
from pathlib import Path

import deepl
import openai
import yaml


class Translator(ABC):
    def __init__(
        self,
        api_key: str,
        input_path: Path,
        source_language: str,
        languages: list[str],
        tag_to_ignore: str | None,
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
        self.source_language = source_language
        self.languages = languages
        self.tag_to_ignore = tag_to_ignore

    @abstractmethod
    def translate(self) -> dict[str, str]:
        """Abstract method that performs translation and returns results."""
        pass

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


class OpenAITranslator(Translator):
    def translate(self, text: str):
        """Translate the given text into multiple languages using the OpenAI API."""
        translated_data = {}
        data = self._load_data()
        text = data["prompt"]

        client = openai.OpenAI()
        openai.api_key = self.api_key

        ignore_text = self._get_ignore_text(tag_to_ignore=self.tag_to_ignore)

        for lang in self.languages:
            prompt = f"""Translate the following text into {lang}{ignore_text}:
            {text}."""
            # Call the API
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a translation assistant."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=100,
            )
            translated_data[lang] = response.choices[0].message

    def _get_ignore_text(self, tag_to_ignore: str | None) -> str:
        if tag_to_ignore is not None:
            closing_tag_to_ignore = re.sub(r"<(\w+)>", r"</\1>", self.tag_to_ignore)
            # Add whitespace at the beginning since the text is appended to the prompt
            ignore_text = (
                f'. Text that is within "{self.tag_to_ignore} {closing_tag_to_ignore}" should not be translated:'
            )
        else:
            ignore_text = ":"
        return ignore_text


class DeepLTranslator(Translator):
    def translate(self) -> dict[str, str]:
        """Translate the given text into multiple languages using the DeepL API."""
        translated_data = {}
        data = self._load_data()
        text = data["prompt"]

        translator = deepl.Translator(self.api_key)
        ignore_tags, tag_handling = (
            self._get_ignore_tags(tag_to_ignore=self.tag_to_ignore) if self.tag_to_ignore is not None else (None, None)
        )

        for lang in self.languages:
            result = translator.translate_text(
                text,
                source_lang=self.source_language,
                target_lang=lang,
                tag_handling=tag_handling,
                ignore_tags=ignore_tags,
            )
            translated_data[lang] = result.text

        return translated_data

    def _get_ignore_tags(self, tag_to_ignore: str) -> tuple[list[str], str]:
        tag_handling = "xml"
        match = re.search(r"<(.*)>", tag_to_ignore)
        if match:
            result = match.group(1)
            ignore_tags = [result]
        else:
            raise ValueError("No tag found in the text.")

        return ignore_tags, tag_handling
