import json
import logging
import statistics
from collections import Counter
from itertools import combinations
from pathlib import Path

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, spearmanr
from sklearn.metrics import cohen_kappa_score, f1_score
from statsmodels.stats.inter_rater import fleiss_kappa

from ml_filter.analysis.plot_score_distributions import plot_confusion_matrix
from ml_filter.analysis.utils import custom_round, get_common_docs, get_document_scores_df
from ml_filter.utils.logging import get_logger

logger = get_logger(name=__name__, level=logging.INFO)  # Set up logging


def prepare_fleiss_data(all_score_pairs: list[list[int]]) -> np.ndarray:
    """
    Prepares data for computing Fleiss' Kappa by transforming scores into a matrix format.

    Args:
        list_of_score_pairs: (list[list[int]]): A list where each sublist contains scores assigned by raters.

    Returns:
        np.ndarray: A 2D matrix where rows correspond to items and columns represent score frequencies.
    """
    max_score = max(max(score_pair) for score_pair in all_score_pairs)
    fleiss_data = np.zeros((len(all_score_pairs), max_score + 1))
    for i, score_pair in enumerate(all_score_pairs):
        for score in score_pair:
            fleiss_data[i, score] += 1
    return fleiss_data


def compute_annotator_correlation(all_score_pairs: list[list[float]], metric: str) -> float:
    """
    Computes the correlation between two annotators' scores.

    Args:
        all_score_pairs (list[list[float]]): List of [annotator1_score, annotator2_score] per item.
        metric (str): Correlation metric: "spearman", "kendall", or "cohen".

    Returns:
        float: Correlation score between the two annotators.
    """
    scores = np.array(all_score_pairs)
    if scores.shape[1] != 2:
        raise ValueError(f"Expected exactly 2 annotators, got {scores.shape[1]}")

    rater1_scores, rater2_scores = scores[:, 0], scores[:, 1]

    if metric == "spearman":
        correlation, _ = spearmanr(rater1_scores, rater2_scores)
    elif metric == "kendall":
        correlation, _ = kendalltau(rater1_scores, rater2_scores)
    elif metric == "cohen":
        correlation = cohen_kappa_score(rater1_scores, rater2_scores)
    else:
        raise ValueError(f"Unsupported metric: {metric}")

    return correlation


# TODO: Remove
# def compute_pairwise_correlations(all_score_pairs: list[list[float]], metric: str) -> float:
#     """
#     Computes the average pairwise correlation between raters' scores.

#     Args:
#         scores (list[list[float]]): A list where each sublist contains scores from all raters for one item.
#         metric (str): The correlation metric to use ("spearman", "kendall", or "cohen").

#     Returns:
#         float: The average pairwise correlation score.
#     """
#     num_annotators = len(all_score_pairs[0])
#     results = []

#     for i in range(num_annotators):
#         for j in range(i + 1, num_annotators):
#             rater1 = [score_pair[i] for score_pair in all_score_pairs]
#             rater2 = [score_pair[j] for score_pair in all_score_pairs]
#             if metric == 'spearman':
#                 correlation, _ = spearmanr(rater1, rater2)
#             elif metric == 'kendall':
#                 correlation, _ = kendalltau(rater1, rater2)
#             elif metric == "cohen":
#                 correlation = cohen_kappa_score(rater1, rater2)
#             results.append(correlation)
#     return np.mean(results)


def compute_krippendorffs_alpha(scores: list[list[float]]) -> float:
    """
    Computes Krippendorff's Alpha, a measure of inter-rater reliability.

    Args:
        scores (list[list[float]]): A list where each sublist contains scores assigned by all raters for one item.

    Returns:
        float: The Krippendorff's Alpha score.
    """
    flattened_scores = np.array(scores).T  # Transpose for Krippendorff's input
    return krippendorff.alpha(reliability_data=flattened_scores, level_of_measurement="ordinal")


def compute_doc_level_variation(all_score_pairs: list[list[int]], document_ids: list[str]) -> dict:
    """
    Computes variation in scores at the document level.

    Args:
        scores (list[list[int]]): A list where each sublist contains scores for a single document.
        document_ids (list[str]): A list of document IDs corresponding to `scores`.

    Returns:
        dict: A dictionary containing variation statistics (mean, standard deviation, counts, etc.).
    """
    score_vars = []
    for score_pair in all_score_pairs:
        score_var = max(score_pair) - min(score_pair)
        score_vars.append(score_var)

    results = {k: v for k, v in zip(document_ids, score_vars)}
    counter = Counter(results.values())
    results["counts"] = {key: counter[key] for key in sorted(counter)}
    results["mean"] = statistics.mean(score_vars)
    results["stdev"] = statistics.stdev(score_vars)

    return results


def compute_gt_metrics(
    ground_truth_scores: list[int],
    predicted_scores: list[int],
    valid_labels: list[int],
) -> dict[str, float]:
    """
    Computes metrics comparing predictions to ground truth.

    Args:
        y_true (list[int]): True values.
        y_pred (list[int]): Predicted values.
        labels (list[int]): The list of possible labels.

    Returns:
        dict[str, float]: A dictionary containing the computed metrics.
    """
    if len(ground_truth_scores) != len(predicted_scores):
        raise ValueError("The number of predictions and labels must be equal.")

    total_num_samples = len(ground_truth_scores)

    # Round labels and predictions
    predictions_rounded = [custom_round(score) for score in predicted_scores]
    ground_truth_rounded = [custom_round(score) for score in ground_truth_scores]

    assert (
        len(predictions_rounded) == total_num_samples
    ), f"Expected {total_num_samples} predictions, got {len(predictions_rounded)}"

    # compute accuracy, mae and mse
    gt_metrics = dict()
    gt_metrics["Acc"] = (
        sum(1 for s1, s2 in zip(ground_truth_rounded, predictions_rounded) if s1 == s2) / total_num_samples
    )
    gt_metrics["MAE"] = sum(abs(a - b) for a, b in zip(ground_truth_scores, predicted_scores)) / total_num_samples
    squared_diffs = [(a - b) ** 2 for a, b in zip(ground_truth_scores, predicted_scores)]
    gt_metrics["MSE"] = sum(squared_diffs) / total_num_samples

    # compute accuracy per class
    class_accuracies = compute_accuracy_per_class(
        ground_truth_scores=ground_truth_rounded, predicted_scores=predictions_rounded, valid_labels=valid_labels
    )
    for valid_label in valid_labels:
        gt_metrics[f"CA-{valid_label}"] = class_accuracies[valid_label]

    # Compute Macro and Micro F1-scores
    gt_metrics["Macro-F1"] = f1_score(ground_truth_rounded, predictions_rounded, average="macro")
    gt_metrics["Micro-F1"] = f1_score(ground_truth_rounded, predictions_rounded, average="micro")

    # Compute F1 score for each class
    class_f1_scores = f1_score(ground_truth_rounded, predictions_rounded, average=None)
    for valid_label, f1 in zip(valid_labels, class_f1_scores):
        gt_metrics[f"F1-{valid_label}"] = f1

    return gt_metrics


def plot_invalid_docs_histogram(
    correct_scores_of_invalid_docs: list[int],
    output_file_path: Path,
    annotator_name: str,
    language: str,
) -> None:
    """
    Plots a histogram of the correct scores for invalid documents.

    Args:
        correct_scores_of_invalid_docs (list[int]): A list of correct scores for invalid documents.
        output_file_path (Path): The path to save the histogram plot.
        annotator_name (str): The name of the annotator.

    Returns:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.hist(correct_scores_of_invalid_docs, bins=[0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5], alpha=0.5, edgecolor="black")
    plt.xlabel("Scores")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of Invalid Scores for {annotator_name} and langauge {language}.")
    plt.grid(True)
    plt.savefig(output_file_path)


def compute_confusion_matrix(
    gt_labels: list[int], valid_labels: list[int], predictions: list[int]
) -> dict[int, dict[int, int]]:
    """
    Computes the confusion matrix for the given ground truth labels and predictions.

    Args:
        gt_labels (list[int]): A list of ground truth labels.
        valid_labels (list[int]): A list of valid labels.
        predictions (list[int]): A list of predicted labels.

    Returns:
        dict[int, dict[int, int]]: A confusion matrix represented as a dictionary.
    """
    predictions = [p if p != "invalid" else -1 for p in predictions]

    # -1 is used for invalid predictions
    all_labels = [-1] + valid_labels
    cm_dict = {valid_label: {label: 0 for label in all_labels} for valid_label in valid_labels}

    for gt_label, prediction in zip(gt_labels, predictions):
        cm_dict[gt_label][prediction] += 1

    return cm_dict


def compute_threshold_agreement(scores: list[tuple[int, int]], threshold: float) -> float:
    """
    Computes the threshold-based agreement between two sets of scores.
    Args:
        scores (list[tuple[int, int]]): A list of tuples containing scores from two annotators.
        threshold (float): The threshold value for agreement.
    Returns:
        float: The threshold-based agreement score.
    """
    above_or_equal_threshold = sum(1 for score_0, score_1 in scores if score_0 >= threshold and score_1 >= threshold)
    below_threshold = sum(1 for score_0, score_1 in scores if score_0 < threshold and score_1 < threshold)
    return (above_or_equal_threshold + below_threshold) / len(scores)


def compute_accuracy_per_class(
    valid_labels: list[int], ground_truth_scores: list[int], predicted_scores: list[int]
) -> dict[int, float]:
    """
    Computes the accuracy for each class based on the ground truth and predicted scores.
    Args:
        valid_labels (list[int]): A list of user-provided valid labels.
        ground_truth_scores (list[int]): A list of ground truth scores.
        predicted_scores (list[int]): A list of predicted scores.
    Returns:
        dict[int, float]: A dictionary where keys are class labels and values are the corresponding accuracies.
    """

    class_accuracies = {}
    for valid_label in valid_labels:
        num_class_samples = sum(1 for score in ground_truth_scores if score == valid_label)
        if num_class_samples == 0:
            class_accuracies[valid_label] = -1.0
            logging.warning(f"No samples for class {valid_label}. Skipping accuracy calculation.")
            continue
        num_correct_predictions = sum(
            1
            for predicted_s, ground_truth_s in zip(predicted_scores, ground_truth_scores)
            if predicted_s == valid_label and ground_truth_s == valid_label
        )
        class_accuracies[valid_label] = num_correct_predictions / num_class_samples
    return class_accuracies


def compute_metrics(num_total_docs: int, valid_docs_df: pd.DataFrame, thresholds: list[float]) -> dict:
    """
    Computes various inter-rater reliability metrics.

    Args:
        num_total_docs (int): The total number of documents.
        valid_docs_df (pd.DataFrame): The DataFrame containing valid document scores.
        thresholds (list[float]): A list of thresholds for computing agreement metrics.

    Returns:
        dict: A dictionary containing the computed metrics.
    """
    # prepare data
    valid_scores = list(zip(valid_docs_df["score_0"], valid_docs_df["score_1"]))
    rounded_valid_scores = list(zip(valid_docs_df["rounded_score_0"], valid_docs_df["rounded_score_1"]))

    # compute metrics
    fleiss_data = prepare_fleiss_data(rounded_valid_scores)
    fk = fleiss_kappa(fleiss_data, method="fleiss")
    spearman_corr = compute_annotator_correlation(valid_scores, metric="spearman")
    kendall_corr = compute_annotator_correlation(valid_scores, metric="kendall")
    cohen_kappa = compute_annotator_correlation(rounded_valid_scores, metric="cohen")
    kripp_alpha = compute_krippendorffs_alpha(valid_scores)
    # TODO: What is the interpretation of this metric?
    # doc_vars = compute_doc_level_variation(rounded_valid_scores, valid_docs_df["doc_id"].tolist())

    # Store results
    metrics = dict()
    metrics["metrics"] = {
        "Fleiss": fk,
        "Cohen": cohen_kappa,
        "Spearman": spearman_corr,
        "Kendall": kendall_corr,
        "Krippendorff": kripp_alpha,
        "Invalid": num_total_docs - len(valid_docs_df),
    }

    # add variation per document
    # metrics["Variation per Document"] = doc_vars
    for threshold in thresholds:
        metrics["metrics"][f"TA-{threshold}"] = compute_threshold_agreement(valid_scores, threshold)

    return metrics


def compare_annotator_to_gt(
    annotators: list[str],
    valid_docs_df: pd.DataFrame,
    common_docs_df: pd.DataFrame,
    metrics: dict,
    output_dir: Path,
    lang: str,
    valid_labels: list[int],
) -> dict:
    """
    Compares annotator annotations to ground truth annotations and computes additional metrics.

    Args:
        annotators (list[str]): A list of annotator names.
        valid_docs_df (pd.DataFrame): The DataFrame containing valid document scores.
        common_docs_df (pd.DataFrame): The DataFrame containing common document scores.
        metrics (dict): A dictionary to store the computed metrics.
        output_dir (Path): The directory to save the output files.
        lang (str): The language of the documents.
        labels (list[int]): The list of possible labels.

    Returns:
        dict: The updated metrics dictionary.
    """
    # in this case there is only one annotator, the other one is the ground truth

    ground_truth_scores = None
    predicted_scores = None

    if annotators[0] == "gt":
        annotator_idx = 1
        gt_idx = 0
        ground_truth_scores = valid_docs_df["score_0"].to_list()
        predicted_scores = valid_docs_df["score_1"].to_list()
    else:
        annotator_idx = 0
        gt_idx = 1
        ground_truth_scores = valid_docs_df["score_1"].to_list()
        predicted_scores = valid_docs_df["score_0"].to_list()

    annotator_name = annotators[annotator_idx]

    gt_metrics = compute_gt_metrics(
        ground_truth_scores=ground_truth_scores,
        predicted_scores=predicted_scores,
        valid_labels=valid_labels,
    )
    metrics["metrics"].update(gt_metrics)

    # plot the distribution of invalid scores
    invalid_docs_df = common_docs_df[common_docs_df[f"score_{annotator_idx}"] == "invalid"]

    # compute confusion matrix
    cm = compute_confusion_matrix(
        gt_labels=common_docs_df[f"rounded_score_{gt_idx}"].to_list(),
        valid_labels=[int(valid_label) for valid_label in valid_labels],
        predictions=common_docs_df[f"rounded_score_{annotator_idx}"].to_list(),
    )

    metrics["CM"] = cm

    # Plot results
    plot_confusion_matrix(
        cm_dict=cm,
        annotator_name=annotator_name,
        output_file_path=output_dir / f"confusion_matrix_{annotator_name}_gt.png",
        valid_labels=[int(valid_label) for valid_label in valid_labels],
        language=lang,
    )

    plot_invalid_docs_histogram(
        correct_scores_of_invalid_docs=invalid_docs_df[f"score_{gt_idx}"].to_list(),
        output_file_path=output_dir / f"histogram_{annotator_name}_invalid_scores.png",
        annotator_name=annotator_name,
        language=lang,
    )

    return metrics


def compute_interrater_reliability_metrics(
    file_paths: list[Path],
    output_dir: Path,
    valid_labels: list[int],
    aggregation_strategy: str,
    thresholds: list[float],
    lang: str,
) -> None:
    """
    Computes various inter-rater reliability metrics and writes results to a JSON file.

    Args:
        file_paths (list[Path, ...]): A list of file paths containing annotation scores in JSONL format.
        output_dir (Path): The output path to save computed metrics as a JSON file.
        valid_labels (list[float]): The list of valid labels.
        aggregation_strategy (str): Specifies how scores for a document from the same file are aggregated.
            Supported values:
            - "mean": Compute the average score.
            - "max": Use the maximum score.
            - "min": Use the minimum score.
            - "majority": Use the score that was voted the most. If there is a tie, take the average of the winners.
        thresholds (list[float]): A list of thresholds for computing agreement metrics.
        lang (str): The language of the documents.

    Raises:
        ValueError: If invalid parameter combinations are provided.

    Returns:
        None
    """
    output_dir.mkdir(exist_ok=True, parents=True)
    document_scores_df = get_document_scores_df(
        input_file_paths=file_paths,
        aggregation_strategy=aggregation_strategy,
        valid_labels=valid_labels,
    )
    metrics = dict()
    annotator_list = list(document_scores_df["annotator"].unique())
    if len(annotator_list) != 2:
        raise ValueError(f"Expected exactly 2 annotators, but found {len(annotator_list)}: {annotator_list}")
    for annotator_0, annotator_1 in combinations(annotator_list, 2):
        # TODO: Do do we deal with the case where for different annotators have different number of documents?
        # filter on documents that are annotated by both annotators and filter out invalid scores
        common_docs_df = get_common_docs(document_scores_df, annotator_0, annotator_1)
        valid_docs_df = common_docs_df[
            (common_docs_df["score_0"] != "invalid") & (common_docs_df["score_1"] != "invalid")
        ]

        # compute metrics
        metrics = compute_metrics(
            num_total_docs=len(common_docs_df),
            valid_docs_df=valid_docs_df,
            thresholds=thresholds,
        )

        # compute additional metrics if one of the annotators is the ground truth
        annotators = [annotator_0, annotator_1]
        if "gt" in annotators:
            metrics = compare_annotator_to_gt(
                annotators=annotators,
                valid_docs_df=valid_docs_df,
                common_docs_df=common_docs_df,
                valid_labels=valid_labels,
                metrics=metrics,
                output_dir=output_dir,
                lang=lang,
            )

        # save results
        output_file_path = output_dir / f"ir_{annotator_0}_{annotator_1}.json"
        with output_file_path.open("w") as f:
            json.dump(metrics, f, indent=4)
