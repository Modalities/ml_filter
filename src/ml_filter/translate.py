import os
from pathlib import Path

import deepl
import yaml


def deepl_translate(api_key: str, input_path: Path, source_language: str, languages: list[str]) -> dict[str, str]:
    translated_data = {}
    with open(input_path, "r") as file:
        data = yaml.safe_load(file)
    text = data["prompt"]

    translator = deepl.Translator(api_key)

    for lang in languages:
        result = translator.translate_text(text, source_lang=source_language, target_lang=lang)
        translated_data[lang] = result.text

    return translated_data


def write_output(output_path: Path, data: dict[str, str]) -> None:
    for lang, text in data.items():
        # Save the data to a YAML file
        with open(os.path.join(output_path, f"educational_prompt_{lang}.yaml"), "w") as file:
            yaml.dump(dict(prompt=text), file, default_flow_style=False, allow_unicode=True)
