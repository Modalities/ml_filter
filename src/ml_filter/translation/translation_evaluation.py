import json
import logging
import os

import numpy as np
from comet import download_model, load_from_checkpoint

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def _load_gold_dict(gold_path: str) -> dict[str, str]:
    """Load reference translations from a JSONL file.

    Args:
        gold_path: Path to the gold reference JSONL file.

    Returns:
        A dictionary mapping document IDs to reference texts.
    """
    gold_dict = {}
    with open(gold_path, "r") as f:
        for line in f:
            item = json.loads(line)
            gold_dict[item["document_id"]] = item["text"]
    return gold_dict


def _prepare_translation_input(file_path: str, gold_dict: dict[str, str]) -> list[dict[str, str]]:
    """Extract source and machine-translated texts from a JSONL file.

    Args:
        file_path: Path to the target JSONL file.
        lang: Language code.
        gold_dict: Dictionary of gold references.

    Returns:
        A list of dictionaries containing 'src' and 'mt' keys.
    """
    target_texts = []
    with open(file_path, "r") as f:
        for line_num, line in enumerate(f, 1):
            if not line:
                continue
            try:
                document = json.loads(line)
                doc_id = document["document_id"]
                text = document["text"]

                if doc_id not in gold_dict:
                    logging.warning(f"doc_id {doc_id} not found in gold references.")
                    continue

                target_texts.append({"src": gold_dict[doc_id], "mt": text})
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid line {line_num} in {file_path}: {e}")
                continue
    return target_texts


def evaluate_translations(
    data_dir: str,
    gold_path: str,
    languages: list[str],
    batch_size: int,
    model_name: str = "Unbabel/wmt22-cometkiwi-da",
) -> None:
    """Evaluate translation quality for a set of files using a COMET model.

    Args:
        data_dir: Directory containing translation JSONL files.
        gold_path: Path to gold reference JSONL file.
        languages: List of supported language codes.
        model_name: COMET model to use.
    """
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    gold_dict = _load_gold_dict(gold_path)
    quality_dict = {}

    for filename in os.listdir(data_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(data_dir, filename)
            lang = filename.split("_")[5]

            if lang not in languages:
                logging.info(f"Skipping file with unsupported language: {file_path}")
                continue

            target_texts = _prepare_translation_input(file_path, gold_dict)

            if target_texts:
                # TODO: ;ultiple GPUs handling
                model_output = model.predict(target_texts, batch_size=batch_size, gpus=1, accelerator="gpu")
                quality_dict[lang] = model_output.scores
                logging.info(f"Processed {len(target_texts)} documents for language '{lang}' in file {file_path}")
            else:
                logging.info(f"No valid documents for language '{lang}' in file {file_path}")

    logging.info("Translation quality scores:")
    for lang, scores in quality_dict.items():
        logging.info(f"Mean score for {lang}: {np.mean(scores):.4f}")
