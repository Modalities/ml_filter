import json
from typing import Dict

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def calculate_classification_metrics(ground_truth_path: str, predictions_path: str) -> Dict:
    """
    Calculate classification metrics by comparing ground truth scores with predictions.

    Args:
        ground_truth_path (str): Path to JSONL file with 'id' and 'score' fields
        predictions_path (str): Path to JSONL file with 'id' and 'prediction' fields

    Returns:
        Dict: Dictionary containing various classification metrics
    """
    # Read and organize the ground truth data
    truth_data = {}
    with open(ground_truth_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            truth_data[entry["id"]] = entry["score"]

    # Read and organize the predictions
    pred_data = {}
    with open(predictions_path, "r") as f:
        for line in f:
            entry = json.loads(line.strip())
            pred_data[entry["id"]] = entry["prediction"]

    # Ensure all IDs match and create ordered lists
    common_ids = set(truth_data.keys()) & set(pred_data.keys())
    if len(common_ids) != len(truth_data) or len(common_ids) != len(pred_data):
        raise ValueError("Mismatch in IDs between ground truth and predictions")

    # Create ordered lists of true values and predictions
    y_true = []
    y_pred = []
    for id_ in sorted(common_ids):  # Sort for reproducibility
        y_true.append(truth_data[id_])
        y_pred.append(pred_data[id_])

    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)

    # Calculate precision, recall, and f1 for each class
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average=None)

    # Calculate macro and weighted averages
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro")

    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )

    # Calculate confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred)

    # Get unique classes
    classes = sorted(set(y_true) | set(y_pred))

    # Prepare per-class metrics
    per_class_metrics = {}
    for i, class_label in enumerate(classes):
        per_class_metrics[class_label] = {
            "precision": precision[i],
            "recall": recall[i],
            "f1": f1[i],
            "support": support[i],
        }

    return {
        "accuracy": accuracy,
        "macro_metrics": {"precision": macro_precision, "recall": macro_recall, "f1": macro_f1},
        "weighted_metrics": {"precision": weighted_precision, "recall": weighted_recall, "f1": weighted_f1},
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": conf_matrix.tolist(),
        "classes": classes,
        "n_samples": len(y_true),
    }


def main():
    metrics = calculate_classification_metrics(
        "../../../data/test_data_ml_filter_511.jsonl", "outputs/output_task0.jsonl"
    )
    print(metrics)


if __name__ == "__main__":
    main()
