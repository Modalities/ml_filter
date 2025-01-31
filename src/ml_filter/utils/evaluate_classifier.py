from typing import List, Dict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, mean_absolute_error


def compute_metrics_for_single_output(
    labels: np.ndarray, preds: np.ndarray, preds_raw: np.ndarray, thresholds: List[int]
) -> Dict[str, float]:
    """
    Computes evaluation metrics for a specific output.

    Args:
        labels (np.ndarray): Ground truth labels of shape (batch_size,)
        preds (np.ndarray): Predicted class indices of shape (batch_size,)
        preds_raw (np.ndarray): Raw predictions (logits or regression values) of shape (batch_size,) for regression
                               and (batch_size, num_classes) for classification.
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

    # Compute classification metrics
    metrics["classification/accuracy"] = accuracy_score(labels, preds)
    metrics["classification/f1_weighted"] = f1_score(labels, preds, average="weighted")
    metrics["classification/f1_micro"] = f1_score(labels, preds, average="micro")
    metrics["classification/f1_macro"] = f1_score(labels, preds, average="macro")

    # Calculate binary metrics for different thresholds
    for threshold in thresholds:
        # Convert to binary predictions using threshold
        binary_preds = np.where(preds >= threshold, 1, 0)
        binary_labels = np.where(labels >= threshold, 1, 0)

        metrics[f"binary/t{threshold}/accuracy"] = accuracy_score(binary_labels, binary_preds)
        metrics[f"binary/t{threshold}/f1_weighted"] = f1_score(binary_labels, binary_preds, average="weighted")
        metrics[f"binary/t{threshold}/f1_micro"] = f1_score(binary_labels, binary_preds, average="micro")
        metrics[f"binary/t{threshold}/f1_macro"] = f1_score(binary_labels, binary_preds, average="macro")

    # Compute regression-like metrics
    metrics["regression/mse"] = mean_squared_error(labels, preds_raw)
    metrics["regression/mae"] = mean_absolute_error(labels, preds_raw)

    # Add f1 scores for each class
    classes = np.unique(labels)
    classes.sort()
    f1_per_class = f1_score(labels, preds, average=None)
    for i, c in enumerate(classes):
        metrics[f"class_f1/f1_class_{c}"] = f1_per_class[i]

    return metrics
