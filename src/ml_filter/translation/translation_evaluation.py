import csv
import json
import logging
import os
from pathlib import Path

import numpy as np
from comet import download_model, load_from_checkpoint
from transformers import AutoTokenizer

from constants import EUROPEAN_LANGUAGES, TRANSLATION_SCORE_CLASSES

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


def _prepare_translation_input(
    file_path: str,
    gold_dict: dict[str, str],
    tokenizer_name_or_path: str,
    max_tokens_per_input: int,
) -> list[dict[str, str]]:
    """Extract source and machine-translated texts from a JSONL file.

    Args:
        file_path: Path to the target JSONL file.
        lang: Language code.
        gold_dict: Dictionary of gold references.

    Returns:
        A list of dictionaries containing 'src' and 'mt' keys.
    """

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    max_src_tokens = max_mt_tokens = max_tokens_per_input // 2

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

                # Tokenize and truncate source and MT
                src_ids = tokenizer.encode(
                    gold_dict[doc_id], truncation=True, max_length=max_src_tokens, add_special_tokens=False
                )
                mt_ids = tokenizer.encode(text, truncation=True, max_length=max_mt_tokens, add_special_tokens=False)

                # Decode back to text
                src_trunc = tokenizer.decode(src_ids, skip_special_tokens=True)
                mt_trunc = tokenizer.decode(mt_ids, skip_special_tokens=True)

                target_texts.append({"src": src_trunc, "mt": mt_trunc})
            except json.JSONDecodeError as e:
                logging.warning(f"Skipping invalid line {line_num} in {file_path}: {e}")
                continue
    return target_texts


def evaluate_translations(
    data_dir: str,
    gold_path: str,
    languages: list[str],
    batch_size: int,
    output_dir: Path,
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
    tokenizer_name_or_path = model.encoder.tokenizer.name_or_path
    max_seq_length = model.encoder.max_positions

    gold_dict = _load_gold_dict(gold_path)
    quality_dict = {}

    for filename in os.listdir(data_dir):
        if filename.endswith(".jsonl"):
            file_path = os.path.join(data_dir, filename)
            parts = filename.split("_")
            if len(parts) != 7:
                logging.warning(f"Skipping file with unexpected format: {file_path}")
                continue
            lang = parts[5]

            if lang not in languages:
                logging.info(f"Skipping file with unsupported language: {file_path}")
                continue

            target_texts = _prepare_translation_input(
                file_path,
                gold_dict,
                tokenizer_name_or_path=tokenizer_name_or_path,
                max_tokens_per_input=max_seq_length,
            )

            if target_texts:
                # TODO: Multiple GPUs handling
                model_output = model.predict(target_texts, batch_size=batch_size, gpus=1, accelerator="gpu")
                quality_dict[lang] = model_output.scores
                logging.info(f"Processed {len(target_texts)} documents for language '{lang}' in file {file_path}")
            else:
                logging.info(f"No valid documents for language '{lang}' in file {file_path}")

    output_path = os.path.join(output_dir, "translation_quality_results.csv")
    _save_to_csv(quality_dict, Path(output_path))


def _save_to_csv(quality_dict: dict[str, list[float]], output_path: Path) -> None:
    """Save translation quality statistics to a CSV file.

    Args:
        quality_dict: Dictionary mapping language code to list of quality scores.
        output_path: Path to save the CSV file.
    """
    with open(output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            ["language", "num_documents", "mean_score", "median_score", "q25_score", "q75_score", "q100_score"]
        )

        for lang, scores in quality_dict.items():
            scores_np = np.array(scores)
            writer.writerow(
                [
                    EUROPEAN_LANGUAGES[lang],
                    len(scores),
                    f"{np.mean(scores_np):.4f}",
                    f"{np.median(scores_np):.4f}",
                    f"{np.quantile(scores_np, 0.25):.4f}",
                    f"{np.quantile(scores_np, 0.75):.4f}",
                    f"{np.max(scores_np):.4f}",
                ]
            )


def _plot_translation_scores_histogram_relative_to_gt(
    id_to_translation_score: dict[str, str],
    id_to_gt_quality_score: dict[str, float],
    lang: str,
    output_path: str,
    counts: dict[float, list[int]],
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    score_classes = ["fine", "minor", "major", "critical"]

    # Step 1: Get all GT scores from the GT dict (not only those that appear in translation dict)
    all_gt_scores = sorted(set(float(v) for v in id_to_gt_quality_score.values()))

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

    ax.set_xlabel("Ground Truth Document Quality Score")
    ax.set_ylabel("Number of Translations")
    ax.set_title(f"Translation Quality vs Ground Truth Document Quality for {EUROPEAN_LANGUAGES[lang]}")
    ax.legend(title="Translation Quality")
    ax.grid(axis="y", alpha=0.5)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def _compute_score_distribution(
    id_to_translation_score: dict[str, str],
    id_to_gt_quality_score: dict[str, float],
) -> dict[float, list[int]]:
    """
    Compute a distribution matrix of translation scores per GT quality score.

    Args:
        id_to_translation_score: Mapping of document ID to translation score label.
        id_to_gt_quality_score: Mapping of document ID to GT quality score (float).
        score_classes: List of allowed translation score class labels in desired order.

    Returns:
        A dict mapping GT quality score (float) to a list of counts aligned with score_classes.
    """

    # Step 1: Get all GT scores from the GT dict (not only those that appear in translation dict)
    all_gt_scores = sorted(set(float(v) for v in id_to_gt_quality_score.values()))

    # Step 2: Build empty counts for all GT scores
    counts = {gt: [0] * len(TRANSLATION_SCORE_CLASSES) for gt in all_gt_scores}

    # Step 3: Count translation scores
    for sample_id, trans_score in id_to_translation_score.items():
        if sample_id in id_to_gt_quality_score:
            gt_score = float(id_to_gt_quality_score[sample_id])
            trans_score = trans_score.lower()
            if trans_score in TRANSLATION_SCORE_CLASSES:
                idx = TRANSLATION_SCORE_CLASSES.index(trans_score)
                counts[gt_score][idx] += 1

    return counts


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
    plt.title(f"Translation Quality Scores for {EUROPEAN_LANGUAGES[lang]}")
    plt.xlabel("Translation Quality")
    plt.ylabel("Frequency")
    plt.grid(axis="y", alpha=0.75)

    # Ensure y-axis ticks are integers
    plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_human_eval_translation_quality_results(
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
    lang_to_eval_stats = {}
    lang_to_counts = {}

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

            counts = _compute_score_distribution(
                id_to_translation_score=id_to_translation_score,
                id_to_gt_quality_score=id_to_gt_quality_score,
            )
            lang_to_counts[lang] = counts

            _plot_translation_scores_histogram_relative_to_gt(
                id_to_translation_score=id_to_translation_score,
                id_to_gt_quality_score=id_to_gt_quality_score,
                lang=lang,
                output_path=output_path,
                counts=counts,
            )
            lang_to_eval_stats[lang] = {
                "num_documents": len(id_to_translation_score),
                "Fine": list(id_to_translation_score.values()).count("fine"),
                "Minor": list(id_to_translation_score.values()).count("minor"),
                "Major": list(id_to_translation_score.values()).count("major"),
                "Critical": list(id_to_translation_score.values()).count("critical"),
            }
    _save_language_eval_stats(
        lang_to_eval_stats=lang_to_eval_stats,
        output_path=os.path.join(output_dir, "language_eval_stats.csv"),
    )

    _save_detailed_score_distribution(
        lang_to_counts=lang_to_counts,
        output_path=os.path.join(output_dir, "detailed_score_distribution.csv"),
    )


def _save_language_eval_stats(lang_to_eval_stats: dict[str, dict], output_path: str) -> None:
    import csv

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "num_documents", "Fine", "Minor", "Major", "Critical"])

        for lang, stats in lang_to_eval_stats.items():
            writer.writerow(
                [
                    lang,
                    stats.get("num_documents", 0),
                    stats.get("Fine", 0),
                    stats.get("Minor", 0),
                    stats.get("Major", 0),
                    stats.get("Critical", 0),
                ]
            )


def _save_detailed_score_distribution(lang_to_counts: dict[str, dict[float, list[int]]], output_path: str) -> None:
    import csv

    with open(output_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["language", "gt_score"] + TRANSLATION_SCORE_CLASSES)

        for lang, counts in lang_to_counts.items():
            for gt_score, count_list in sorted(counts.items()):
                writer.writerow([lang, f"{gt_score:.2f}"] + count_list)
