
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import Dict, List, Tuple, Optional

import krippendorff
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from statsmodels.stats.inter_rater import fleiss_kappa

from ml_filter.analysis.utils import get_document_scores


def prepare_fleiss_data(all_scores: List[List[int]]) -> np.ndarray:
    """
    Prepares data for computing Fleiss' Kappa by transforming scores into a matrix format.

    Args:
        all_scores (List[List[int]]): A list where each sublist contains scores assigned by raters.

    Returns:
        np.ndarray: A 2D matrix where rows correspond to items and columns represent score frequencies.
    """
    max_score = max(max(scores) for scores in all_scores)
    fleiss_data = np.zeros((len(all_scores), max_score + 1))
    for i, scores in enumerate(all_scores):
        for score in scores:
            fleiss_data[i, score] += 1
    return fleiss_data


def compute_pairwise_correlations(all_scores: List[List[float]], metric: str) -> float:
    """
    Computes the average pairwise correlation between raters' scores.

    Args:
        all_scores (List[List[float]]): A list where each sublist contains scores from all raters for one item.
        metric (str): The correlation metric to use ("spearman", "kendall", or "cohen").

    Returns:
        float: The average pairwise correlation score.
    """
    n = len(all_scores[0])
    results = []
    for i in range(n):
        for j in range(i + 1, n):
            rater1 = [scores[i] for scores in all_scores]
            rater2 = [scores[j] for scores in all_scores]
            if metric == 'spearman':
                correlation, _ = spearmanr(rater1, rater2)
            elif metric == 'kendall':
                correlation, _ = kendalltau(rater1, rater2)
            elif metric == "cohen":
                correlation = cohen_kappa_score(rater1, rater2)
            results.append(correlation)
    return np.mean(results)


def compute_krippendorffs_alpha(all_scores: List[List[float]]) -> float:
    """
    Computes Krippendorff's Alpha, a measure of inter-rater reliability.

    Args:
        all_scores (List[List[float]]): A list where each sublist contains scores assigned by all raters for one item.

    Returns:
        float: The Krippendorff's Alpha score.
    """
    flattened_scores = np.array(all_scores).T  # Transpose for Krippendorff's input
    return krippendorff.alpha(reliability_data=flattened_scores, level_of_measurement='ordinal')


def compute_doc_level_variation(all_scores: List[List[int]], all_document_ids: List[str]) -> Dict:
    """
    Computes variation in scores at the document level.

    Args:
        all_scores (List[List[int]]): A list where each sublist contains scores for a single document.
        all_document_ids (List[str]): A list of document IDs corresponding to `all_scores`.

    Returns:
        dict: A dictionary containing variation statistics (mean, standard deviation, counts, etc.).
    """
    score_vars = []
    for scores in all_scores:
        score_var = max(scores) - min(scores)
        score_vars.append(score_var)
        
    results = {k: v for k, v in zip(all_document_ids, score_vars)}
    counter = Counter(results.values())
    results["counts"] = {key: counter[key] for key in sorted(counter)}
    results["mean"] = statistics.mean(score_vars)
    results["stdev"] = statistics.stdev(score_vars)

    return results


def compute_average_accuracy_mae_mse_against_gt(all_scores: List[List[int]], all_scores_rounded: List[List[int]], gt_file_idx: int) -> Tuple[float, float]:
    """
    Computes the accuracy of the annotators' scores against the ground truth.

    Args:
        all_scores (List[List[int]]): A list where each sublist contains scores assigned by all raters for one item.
        gt_file_idx (int): The index of the ground truth file in the list of all files.

    Returns:
        None
    """
    gt_scores = [item[gt_file_idx] for item in all_scores]
    gt_scores_rounded = [item[gt_file_idx] for item in all_scores_rounded]
    num_annotators = len(all_scores[0]) - 1
    total_acc = 0
    total_mae = 0
    total_mse = 0
    for i in range(len(all_scores[0])):
        if i == gt_file_idx:
            continue
        annotator_scores = [item[i] for item in all_scores]
        annotator_scores_rounded = [item[i] for item in all_scores_rounded]
        acc = sum(1 for s1, s2 in zip(gt_scores_rounded, annotator_scores_rounded) if s1 == s2) / len(gt_scores_rounded)
        total_acc += acc
        mae = sum(abs(a - b) for a, b in zip(gt_scores, annotator_scores)) / len(gt_scores)
        total_mae += mae
        squared_diffs = [(a - b) ** 2 for a, b in zip(gt_scores, annotator_scores)]
        mse = sum(squared_diffs) / len(squared_diffs)
        total_mse += mse
    
    avg_metrics = {
        'acc': total_acc / num_annotators,
        'mae': total_mae / num_annotators,
        'mse': total_mse / num_annotators,
    }
    return avg_metrics
    
    
def plot_histogram(missing_scores: Dict[str, List[int]], gt_file_idx: int, output_file_path: str, model_name: str) -> None:
    # Plot the histogram for missing scores
    plt.figure(figsize=(10, 6))
    plt.hist(
        [scores[gt_file_idx] for scores in missing_scores.values()],
        bins=[0, 0.5, 1.5, 2.5, 3.5, 4.5, 5.5],
        alpha=0.5,
        edgecolor='black'
    )
    plt.xlabel('Scores')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Invalid Scores for {model_name}')
    plt.grid(True)
    plt.savefig(output_file_path)
    
    
def plot_confusion_matrix(labels: List[int], preds: List[int], output_file_path: str, model_name: str) -> None:
    # Plot the confusion matrix for missing scores
    preds = [p if p != "invalid" else -1 for p in preds]
    cm = confusion_matrix(labels, preds)
    
    # Normalize the confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Plot the confusion matrix
    plt.figure(figsize=(10, 6))
    xlabels = [f'{pred}' if pred != -1 else "invalid" for pred in np.unique(preds)]
    if "invalid" in xlabels:
        # drop row for label "invalid", as it is not a gold label
        cm = np.delete(cm, 0, axis=0)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', xticklabels=xlabels, yticklabels=np.unique(labels))
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.savefig(output_file_path)
    plt.show()
    
    cm_dict = {int(label): {int(pred): int(cm[i, j]) for j, pred in enumerate(np.unique(preds))} for i, label in enumerate(np.unique(labels))}
    return cm_dict

    
def compute_interrater_reliability_metrics(
    path_to_files: Tuple[Path],
    output_dir: Path,
    model_name: str,
    aggregation: Optional[str] = None,
    gt_file_idx: Optional[int] = None,
    max_score: Optional[int] = None,
) -> None:
    """
    Computes various inter-rater reliability metrics and writes results to a JSON file. 
    
    The different annotators can be placed in separate files or in a single file. 
    In the first case, the scores for each document in each file have to be aggregated first, as they all represent the same annotator.
    In the second case, there should no aggregation happen, which is specified by setting the parameter "aggregation" to None.

    Args:
        path_to_files (Tuple[Path]): A tuple of file paths containing annotation scores in JSONL format.
        output_file_path (Path): The output path to save computed metrics as a JSON file.
        aggregation (Optional[str], optional): Specifies how scores for a document from the same file are aggregated.
            Supported values:
            - "mean": Compute the average score.
            - "max": Use the maximum score.
            - "min": Use the minimum score.
            - "majority": Use the score that was voted the most. If there is a tie, take the average of the winners.
            - None: No aggregation (used for individual annotator analysis).

    Raises:
        ValueError: If invalid parameter combinations are provided.

    Returns:
        None
    """    
    document_scores = get_document_scores(
        path_to_files=path_to_files,
        aggregation=aggregation,
        max_score=max_score
    )
    metrics = {}
    for prompt in document_scores:
        all_document_ids = []
        all_scores = []
        missing_scores = {}
        
        num_versions = max(len(versions) for versions in document_scores[prompt].values())
        for document_id, scores_per_version in document_scores[prompt].items():
            scores = []
            for version in scores_per_version:
                scores.append(scores_per_version[version])

            # Skip documents where not all versions have scores
            if len(scores) != num_versions:
                continue
            
            if "invalid" in scores:
                missing_scores[document_id] = scores
                continue
            
            all_scores.append(scores)
            all_document_ids.append(document_id)

        # Metrics for rounded scores
        all_scores_rounded = [[round(val) for val in scores] for scores in all_scores]
        
        # Compute metrics
        fleiss_data = prepare_fleiss_data(all_scores_rounded)
        fk = fleiss_kappa(fleiss_data, method='fleiss')
        spearman_corr = compute_pairwise_correlations(all_scores, metric='spearman')
        kendall_corr = compute_pairwise_correlations(all_scores, metric='kendall')
        cohen_kappa = compute_pairwise_correlations(all_scores_rounded, metric='cohen')
        kripp_alpha = compute_krippendorffs_alpha(all_scores)
        doc_vars = compute_doc_level_variation(all_scores_rounded, all_document_ids)
        num_invalid_scores = len(missing_scores)
        
        # Store results
        metrics[prompt] = {
            'Fleiss Kappa': fk,
            'Cohen Kappa (avg pairwise)': cohen_kappa,
            'Spearman Rank Correlation (avg pairwise)': spearman_corr,
            'Kendall Tau (avg pairwise)': kendall_corr,
            'Krippendorff Alpha': kripp_alpha,
            "Variation per Document": doc_vars,
            "Number of invalid scores": num_invalid_scores
        }
        
        # compute accuracy and mse if ground truth is provided
        if gt_file_idx is not None:
            avg_metrics = compute_average_accuracy_mae_mse_against_gt(
                all_scores=all_scores, 
                all_scores_rounded=all_scores_rounded,
                gt_file_idx=gt_file_idx
            )
            metrics[prompt]['Accuracy against GT (avg pairwise)'] = avg_metrics["acc"]
            metrics[prompt]['MAE against GT (avg pairwise)'] = avg_metrics["mae"]
            metrics[prompt]['MSE against GT (avg pairwise)'] = avg_metrics["mse"]
            
            # plot the distribution of invalid scores
            plot_histogram(
                missing_scores=missing_scores,
                gt_file_idx=gt_file_idx,
                output_file_path=output_dir / f"histogram_{prompt}_{model_name}.png",
                model_name=model_name,
            )
            
            # compute confusion matrix
            labels = []
            preds = []
            for scores in all_scores_rounded + list(missing_scores.values()):
                if len(scores) != 2:
                    raise ValueError("Confusion matrix can only be computed for two annotators.")
                
                for i, score in enumerate(scores):
                    if i == gt_file_idx:
                        label = score
                    else:
                        pred = score
                # convert scores from missing scores to integers
                labels.append(int(label))
                if pred != "invalid":
                    pred = int(pred)
                preds.append(pred)
                
            cm = plot_confusion_matrix(
                labels=labels,
                preds=preds,
                output_file_path=output_dir / f"confusion_matrix_{prompt}_{model_name}.png",
                model_name=model_name,
            )
            metrics[prompt]['CM against GT'] = cm

    # Print results and save them to file
    print("\n".join(f"{key}: {value}" for key, value in metrics.items()))
    output_file_path = output_dir / f"ir_{model_name}_gt.json"
    with output_file_path.open("w") as f:
        json.dump(metrics, f, indent=4)
