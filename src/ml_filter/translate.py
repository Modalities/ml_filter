import os
import re
from pathlib import Path

import deepl
import yaml


def deepl_translate(
    api_key: str, input_path: Path, source_language: str, languages: list[str], tag_to_ignore: str | None
) -> dict[str, str]:
    translated_data = {}
    with open(input_path, "r") as file:
        data = yaml.safe_load(file)
    text = data["prompt"]

    translator = deepl.Translator(api_key)

    if tag_to_ignore is None:
        ignore_tags = None
        tag_handling = None
    else:
        tag_handling = "xml"
        match = re.search(r"<(.*)>", tag_to_ignore)
        if match:
            result = match.group(1)
        else:
            raise ValueError("No tag found in the text.")
        ignore_tags = [result]

    for lang in languages:
        result = translator.translate_text(
            text, source_lang=source_language, target_lang=lang, tag_handling=tag_handling, ignore_tags=ignore_tags
        )
        translated_data[lang] = result.text

    return translated_data


def write_output(output_path: Path, data: dict[str, str]) -> None:
    for lang, text in data.items():
        # Save the data to a YAML file
        with open(os.path.join(output_path, f"educational_prompt_{lang}.yaml"), "w") as file:
            yaml.dump(dict(prompt=text), file, default_flow_style=False, allow_unicode=True)
