from collections import Counter
from itertools import combinations
import json
import logging
from pathlib import Path
import statistics

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score, f1_score
from statsmodels.stats.inter_rater import fleiss_kappa

from ml_filter.analysis.utils import get_common_docs, get_document_scores
from ml_filter.utils.logging import get_logger


logger = get_logger(name=__name__, level=logging.INFO) # Set up logging


def prepare_fleiss_data(scores: list[list[int]]) -> np.ndarray:
    """
    Prepares data for computing Fleiss' Kappa by transforming scores into a matrix format.

    Args:
        scores (list[list[int]]): A list where each sublist contains scores assigned by raters.

    Returns:
        np.ndarray: A 2D matrix where rows correspond to items and columns represent score frequencies.
    """
    max_score = max(max(scores) for scores in scores)
    fleiss_data = np.zeros((len(scores), max_score + 1))
    for i, scores in enumerate(scores):
        for score in scores:
            fleiss_data[i, score] += 1
    return fleiss_data


def compute_pairwise_correlations(scores: list[list[float]], metric: str) -> float:
    """
    Computes the average pairwise correlation between raters' scores.

    Args:
        scores (list[list[float]]): A list where each sublist contains scores from all raters for one item.
        metric (str): The correlation metric to use ("spearman", "kendall", or "cohen").

    Returns:
        float: The average pairwise correlation score.
    """
    n = len(scores[0])
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            rater1 = [scores[i] for scores in scores]
            rater2 = [scores[j] for scores in scores]
            if metric == 'spearman':
                correlation, _ = spearmanr(rater1, rater2)
            elif metric == 'kendall':
                correlation, _ = kendalltau(rater1, rater2)
            elif metric == "cohen":
                correlation = cohen_kappa_score(rater1, rater2)
            results.append(correlation)
    return np.mean(results)


def compute_krippendorffs_alpha(scores: list[list[float]]) -> float:
    """
    Computes Krippendorff's Alpha, a measure of inter-rater reliability.

    Args:
        scores (list[list[float]]): A list where each sublist contains scores assigned by all raters for one item.

    Returns:
        float: The Krippendorff's Alpha score.
    """
    flattened_scores = np.array(scores).T  # Transpose for Krippendorff's input
    return krippendorff.alpha(reliability_data=flattened_scores, level_of_measurement='ordinal')


def compute_doc_level_variation(scores: list[list[int]], document_ids: list[str]) -> dict:
    """
    Computes variation in scores at the document level.

    Args:
        scores (list[list[int]]): A list where each sublist contains scores for a single document.
        document_ids (list[str]): A list of document IDs corresponding to `scores`.

    Returns:
        dict: A dictionary containing variation statistics (mean, standard deviation, counts, etc.).
    """
    score_vars = []
    for scores in scores:
        score_var = max(scores) - min(scores)
        score_vars.append(score_var)
        
    results = {k: v for k, v in zip(document_ids, score_vars)}
    counter = Counter(results.values())
    results["counts"] = {key: counter[key] for key in sorted(counter)}
    results["mean"] = statistics.mean(score_vars)
    results["stdev"] = statistics.stdev(score_vars)

    return results


def compute_gt_metrics(
    y_true: list[int],
    y_pred: list[int],
    labels: list[int],
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
    if len(y_true) != len(y_pred):
        raise ValueError("The number of predictions and labels must be equal.")
    
    # Round labels and predictions
    y_pred_rounded = [round(score) for score in y_pred]
    y_true_rounded = [round(score) for score in y_true] 
    
    # compute accuracy, mae and mse
    gt_metrics = dict()
    gt_metrics["Acc"] = sum(1 for s1, s2 in zip(y_true_rounded, y_pred_rounded) if s1 == s2) / len(y_pred_rounded)
    gt_metrics["MAE"] = sum(abs(a - b) for a, b in zip(y_true, y_pred)) / len(y_pred)
    squared_diffs = [(a - b) ** 2 for a, b in zip(y_true, y_pred)]
    gt_metrics["MSE"] = sum(squared_diffs) / len(squared_diffs)
    
    # compute accuracy per class
    class_accuracies = compute_accuracy_per_class(
        y_true=y_true_rounded,
        y_pred=y_pred_rounded,
        labels=labels
    )
    for c in labels:
        gt_metrics[f"CA-{c}"] = class_accuracies[c]    
        
    # Compute Macro and Micro F1-scores
    gt_metrics["Macro-F1"] = f1_score(y_true_rounded, y_pred_rounded, average="macro")
    gt_metrics["Micro-F1"] = f1_score(y_true_rounded, y_pred_rounded, average="micro")

    # Compute F1 score for each class
    class_f1_scores = f1_score(y_true_rounded, y_pred_rounded, average=None)
    for c, f1 in zip(labels, class_f1_scores):
        gt_metrics[f"F1-{c}"] = f1
        
    return gt_metrics
    
    
def plot_invalid_docs_histogram(
    correct_scores_of_invalid_docs: list[int],
    output_file_path: Path,
    annotator_name: str
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
    plt.hist(
        correct_scores_of_invalid_docs,
        bins=[0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        alpha=0.5,
        edgecolor='black'
    )
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Invalid Scores for {annotator_name}')
    plt.grid(True)
    plt.savefig(output_file_path)
    
    
def compute_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    output_file_path: Path,
    annotator_name: str,
    lang: str,
) -> dict[int, dict[int, int]]:
    """
    Computes and plots the confusion matrix for the given labels and predictions.

    Args:
        y_true (list[int]): The ground truth labels.
        y_pred (list[int]): The predicted labels.
        output_file_path (Path): The path to save the confusion matrix plot.
        annotator_name (str): The name of the annotator.
        lang (str): The language of the documents.

    Returns:
        dict[int, dict[int, int]]: The confusion matrix as a dictionary.
    """
    y_pred = [p if p != "invalid" else -1 for p in y_pred]
    
    label_classes = list(range(6))
    pred_classes = [-1] + label_classes
    cm_dict = {label: {pred: 0 for pred in pred_classes} for label in label_classes}
    
    for l, p in zip(y_true, y_pred):
        cm_dict[l][p] += 1
    
    # Convert cm_dict to a 2D list
    cm_array = [[cm_dict[label][pred] for pred in pred_classes] for label in label_classes]

    # Convert the 2D list to a NumPy array
    cm_array = np.array(cm_array)
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 6))
        
    # Normalize the confusion matrix
    cm_normalized = cm_array.astype('float') / cm_array.sum(axis=1)[:, np.newaxis]
    xlabels = [p if p != -1 else "invalid" for p in pred_classes]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=xlabels, yticklabels=label_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {annotator_name} and language {lang}')
    plt.savefig(output_file_path)
    plt.show()
    
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
    above_threshold = sum(
        1 for score_0, score_1 in scores if score_0 > threshold and score_1 > threshold
    )
    below_threshold = sum(
        1 for score_0, score_1 in scores if score_0 <= threshold and score_1 <= threshold
    )
    return (above_threshold + below_threshold) / len(scores)


def compute_accuracy_per_class(
    labels: list[int],
    y_true: list[int],
    y_pred: list[int]
) -> dict[int, float]:
    """
    Computes the accuracy per class for the given scores.
    Args:
        labels (list[int]): The list of possible labels.
        y_true (list[int]): True values.
        y_pred (list[int]): Predicted values.
    Returns:
        dict: A dictionary containing the accuracy for each class.
    """
    class_accuracies = {}
    for c in labels:
        total = sum(1 for score in y_true if score == c)
        correct = sum(1 for p, t in zip(y_pred, y_true) if p == c and t == c)
        class_accuracies[c] = correct / total if total > 0 else 0.0
    return class_accuracies


def compute_metrics(
    num_total_docs: int,
    valid_docs_df: pd.DataFrame,
    thresholds: list[float]
) -> dict:
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
    fk = fleiss_kappa(fleiss_data, method='fleiss')
    spearman_corr = compute_pairwise_correlations(valid_scores, metric='spearman')
    kendall_corr = compute_pairwise_correlations(valid_scores, metric='kendall')
    cohen_kappa = compute_pairwise_correlations(rounded_valid_scores, metric='cohen')
    kripp_alpha = compute_krippendorffs_alpha(valid_scores)
    doc_vars = compute_doc_level_variation(rounded_valid_scores, valid_docs_df["doc_id"].tolist())
     
    # Store results
    metrics = dict()
    metrics["metrics"] = {
        'Fleiss': fk,
        'Cohen': cohen_kappa,
        'Spearman': spearman_corr,
        'Kendall': kendall_corr,
        'Krippendorff': kripp_alpha,
        'Invalid': num_total_docs - len(valid_docs_df),
    }
    
    # add variation per document
    metrics["Variation per Document"] = doc_vars
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
    labels: list[int],
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
    if annotators[0] == "gt":
        annotator_idx = 1
        gt_idx = 0
        y_true = valid_docs_df["score_0"].to_list()
        y_pred = valid_docs_df["score_1"].to_list()
    else:
        annotator_idx = 0
        gt_idx = 1
        y_true = valid_docs_df["score_1"].to_list()
        y_pred = valid_docs_df["score_0"].to_list()
    annotator_name = annotators[annotator_idx]
    
    # compute accuracy, mae and mse against ground truth
    gt_metrics = compute_gt_metrics(
        y_true=y_true, 
        y_pred=y_pred,
        labels=labels,
    )
    metrics["metrics"].update(gt_metrics)  
    
    # plot the distribution of invalid scores
    invalid_docs_df = common_docs_df[common_docs_df[f"score_{annotator_idx}"] == "invalid"]
    plot_invalid_docs_histogram(
        correct_scores_of_invalid_docs=invalid_docs_df[f"score_{gt_idx}"].to_list(),
        output_file_path=output_dir / f"histogram_{annotator_name}_invalid_scores.png",
        annotator_name=annotator_name
    )
    
    # compute confusion matrix                
    cm = compute_confusion_matrix(
        y_true=common_docs_df[f"rounded_score_{gt_idx}"].to_list(),
        y_pred=common_docs_df[f"rounded_score_{annotator_idx}"].to_list(),
        output_file_path=output_dir / f"confusion_matrix_{annotator_name}_gt.png",
        annotator_name=annotator_name,
        lang=lang,
    )
    metrics['CM'] = cm
    
    return metrics

    
def compute_interrater_reliability_metrics(
    file_paths: tuple[Path, ...],
    output_dir: Path,
    labels: list[int],
    aggregation: str,
    thresholds: list[float],
    lang: str,
) -> None:
    """
    Computes various inter-rater reliability metrics and writes results to a JSON file. 
    
    Args:
        file_paths (tuple[Path, ...]): A tuple of file paths containing annotation scores in JSONL format.
        output_dir (Path): The output path to save computed metrics as a JSON file.
        labels (list[int]): The list of possible labels.
        aggregation (str): Specifies how scores for a document from the same file are aggregated.
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
    document_scores_df = get_document_scores(
        file_paths=file_paths,
        aggregation=aggregation,
        labels=labels,
    )
    metrics = dict()
    for annotator_0, annotator_1 in combinations(document_scores_df["annotator"].unique(), 2):
        # filter on documents that are annotated by both annotators and filter out invalid scores
        common_docs_df = get_common_docs(document_scores_df, annotator_0, annotator_1)
        valid_docs_df = common_docs_df[(common_docs_df["score_0"] != "invalid") & (common_docs_df["score_1"] != "invalid")]
        
        # compute metrics
        metrics = compute_metrics(
            num_total_docs=len(common_docs_df),
            valid_docs_df=valid_docs_df,
            thresholds=thresholds
        )
        
        # compute additional metrics if one of the annotators is the ground truth
        annotators = [annotator_0, annotator_1]
        if "gt" in annotators:
            metrics = compare_annotator_to_gt(
                annotators=annotators,
                valid_docs_df=valid_docs_df,
                common_docs_df=common_docs_df,
                metrics=metrics,
                output_dir=output_dir,
                lang=lang,
                labels=labels,
            )

        # save results
        output_file_path = output_dir / f"ir_{annotator_0}_{annotator_1}.json"
        with output_file_path.open("w") as f:
            json.dump(metrics, f, indent=4)
