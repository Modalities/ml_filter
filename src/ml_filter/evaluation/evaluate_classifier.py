import logging
from typing import Dict, List

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error


def compute_metrics_for_single_output(
    labels: np.ndarray, predictions: np.ndarray, predictions_raw: np.ndarray, thresholds: List[int]
) -> Dict[str, float]:
    """
    Computes evaluation metrics for a specific output.

    Args:
        labels (np.ndarray): Ground truth labels of shape (batch_size,)
        predictions (np.ndarray): Predicted class indices of shape (batch_size,)
        predictions_raw (np.ndarray): Raw predictions (logits or regression values) of
          shape (batch_size,) for regression and (batch_size, num_classes) for classification.
        thresholds (list): List of thresholds to use for binary metrics

    Returns:
        dict: Dictionary containing the following metrics:
            - accuracy: Overall classification accuracy
            - f1_weighted: F1 score with weighted averaging
            - f1_micro: F1 score with micro averaging
            - f1_macro: F1 score with macro averaging
            - binary_accuracy_t{t}: Binary accuracy for each threshold t
            - binary_f1_weighted_t{t}: Binary F1 weighted for each threshold t
            - binary_f1_micro_t{t}: Binary F1 micro for each threshold t
            - binary_f1_macro_t{t}: Binary F1 macro for each threshold t
            - mse: Mean squared error between raw predictions and labels
            - mae: Mean absolute error between raw predictions and labels
            - f1_class_{c}: F1 score for each individual class c
    """
    metrics = {}

    # Check if labels are continuous (contain non-integer values)
    labels_are_continuous = not np.all(labels == labels.astype(int))

    if labels_are_continuous:
        logging.warning("Detected continuous labels - skipping classification metrics")
    else:
        logging.info(f"Detected discrete labels with {len(np.unique(labels))} unique classes")

    # Compute classification metrics only if labels appear to be discrete
    if not labels_are_continuous:
        metrics["classification/accuracy"] = accuracy_score(labels, predictions)
        metrics["classification/f1_weighted"] = f1_score(labels, predictions, average="weighted")
        metrics["classification/f1_micro"] = f1_score(labels, predictions, average="micro")
        metrics["classification/f1_macro"] = f1_score(labels, predictions, average="macro")

    # Calculate binary metrics for different thresholds
    for threshold in thresholds:
        # Convert to binary predictions using threshold
        binary_preds = np.where(predictions >= threshold, 1, 0)
        binary_labels = np.where(labels >= threshold, 1, 0)

        metrics[f"binary/t{threshold}/accuracy"] = accuracy_score(binary_labels, binary_preds)
        metrics[f"binary/t{threshold}/f1_weighted"] = f1_score(binary_labels, binary_preds, average="weighted")
        metrics[f"binary/t{threshold}/f1_micro"] = f1_score(binary_labels, binary_preds, average="micro")
        metrics[f"binary/t{threshold}/f1_macro"] = f1_score(binary_labels, binary_preds, average="macro")

    # Compute regression-like metrics
    metrics["regression/mse"] = mean_squared_error(labels, predictions_raw)
    metrics["regression/mae"] = mean_absolute_error(labels, predictions_raw)
    metrics["spearman_corr"], _ = spearmanr(predictions_raw, labels)

    # Add f1 scores for each class only if labels are discrete
    if not labels_are_continuous:
        classes = np.unique(labels)
        classes.sort()
        f1_per_class = f1_score(labels, predictions, average=None)
        for i, c in enumerate(classes):
            metrics[f"class_f1/f1_class_{c}"] = f1_per_class[i]

    logging.info(f"Computed {len(metrics)} total metrics")
    return metrics
