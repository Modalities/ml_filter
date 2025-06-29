import json
import logging
import os
from pathlib import Path

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
            parts = filename.split("_")
            if len(parts) != 8:
                logging.warning(f"Skipping file with unexpected format: {file_path}")
                continue
            lang = parts[5]

            if lang not in languages:
                logging.info(f"Skipping file with unsupported language: {file_path}")
                continue

            target_texts = _prepare_translation_input(file_path, gold_dict)

            if target_texts:
                # TODO: Multiple GPUs handling
                model_output = model.predict(target_texts, batch_size=batch_size, gpus=1, accelerator="gpu")
                quality_dict[lang] = model_output.scores
                logging.info(f"Processed {len(target_texts)} documents for language '{lang}' in file {file_path}")
            else:
                logging.info(f"No valid documents for language '{lang}' in file {file_path}")

    logging.info("Translation quality scores:")
    for lang, scores in quality_dict.items():
        logging.info(f"Mean score for {lang}: {np.mean(scores):.4f}")


def _plot_translation_scores_histogram_relative_to_gt(
    id_to_translation_score: dict[str, str], id_to_gt_quality_score: dict[str, float], lang: str, output_path: str
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    score_classes = ["fine", "minor", "major", "critical"]

    # Step 1: Get all GT scores from the GT dict (not only those that appear in translation dict)
    all_gt_scores = sorted(set(float(v) for v in id_to_gt_quality_score.values()))

    # Step 2: Build empty counts for all GT scores
    counts = {gt: [0] * len(score_classes) for gt in all_gt_scores}

    # Step 3: Count translation scores
    for sample_id, trans_score in id_to_translation_score.items():
        if sample_id in id_to_gt_quality_score:
            gt_score = float(id_to_gt_quality_score[sample_id])
            trans_score = trans_score.lower()
            if trans_score in score_classes:
                idx = score_classes.index(trans_score)
                counts[gt_score][idx] += 1

    # Step 4: Plot
    x = np.array(all_gt_scores)
    bar_width = 0.12
    fig, ax = plt.subplots()

    for i, score_class in enumerate(score_classes):
        offsets = x + i * bar_width
        heights = [counts[gt][i] for gt in all_gt_scores]
        ax.bar(offsets, heights, width=bar_width, label=score_class.capitalize())

    # X-ticks in center of bar groups
    tick_positions = x + bar_width * (len(score_classes) - 1) / 2
    tick_labels = [str(int(v)) if v.is_integer() else "" for v in x]

    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels)

    ax.set_xlabel("Ground Truth Quality Score")
    ax.set_ylabel("Number of Translations")
    ax.set_title(f"Translation Quality vs Ground Truth Quality for {lang}")
    ax.legend(title="Translation Score Class")
    ax.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _plot_translation_scores_histogram(scores: list[str], lang: str, output_path: str) -> None:
    """Plot a histogram of translation quality scores and save it to a file.

    Args:
        scores: List of quality scores.
        lang: Language code for the histogram title.
        output_path: Path to save the histogram figure.
    """
    from collections import Counter

    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    # Enforce lowercase and fixed order
    score_classes = ["fine", "minor", "major", "critical"]
    scores = [s.lower() for s in scores]
    counts = Counter(scores)
    values = [counts[cls] for cls in score_classes]

    plt.bar(score_classes, values, alpha=0.7)
    plt.title(f"Translation Quality Scores for {lang}")
    plt.xlabel("Translation Score")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    # Ensure y-axis ticks are integers
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_translation_quality_results(
    data_dir: Path,
    gt_path: Path,
    languages: list[str],
    output_dir: Path,
) -> None:
    """Plot histograms for translation quality results.

    Args:
        data_dir: Directory containing translation JSONL files.
        languages: List of supported language codes.
        output_dir: Directory to save the histogram plots.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    id_to_gt_quality_score = {}

    with open(gt_path, "r") as f:
        for line in f:
            item = json.loads(line)
            id_to_gt_quality_score[item["document_id"]] = float(item["score"])

    for filename in os.listdir(data_dir):
        if filename.endswith(".json"):
            parts = filename.split("_")
            if len(parts) != 8:
                continue
            lang = parts[5]

            if lang not in languages:
                logging.warning(f"Skipping file with unsupported language: {filename}")
                continue

            file_path = os.path.join(data_dir, filename)
            id_to_translation_score = {}
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)  # Load the full list
                for item in data:
                    id_to_translation_score[item["document_id"]] = item["translation_score"].lower()

            output_path = os.path.join(output_dir, f"{lang}_translation_quality_histogram.png")
            _plot_translation_scores_histogram(
                scores=list(id_to_translation_score.values()),
                lang=lang,
                output_path=output_path,
            )

            output_path = os.path.join(output_dir, f"{lang}_translation_quality_vs_gt_histogram.png")
            _plot_translation_scores_histogram_relative_to_gt(
                id_to_translation_score=id_to_translation_score,
                id_to_gt_quality_score=id_to_gt_quality_score,
                lang=lang,
                output_path=output_path,
            )
