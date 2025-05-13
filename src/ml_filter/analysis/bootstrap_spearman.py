from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import itertools

from ml_filter.analysis.utils import get_document_scores_df

def bootstrap_spearman_multi(preds_dict, ground_truth, n_bootstrap=10000, seed=42):
    """
    Computes bootstrap estimates of pairwise differences in Spearman correlation
    between multiple models and the ground truth.

    Args:
        preds_dict: dict of model_name -> prediction array (length N)
        ground_truth: array of length N
        n_bootstrap: number of bootstrap iterations
        seed: random seed

    Returns:
        A dict of results:
            - mean_diffs: dict of (model_i, model_j) -> mean difference
            - ci: dict of (model_i, model_j) -> (lower, upper)
            - p_values: dict of (model_i, model_j) -> two-sided p-value
    """
    rng = np.random.default_rng(seed)
    model_names = list(preds_dict.keys())

    # Store per-iteration rho for each model
    rhos = {name: [] for name in model_names}

    # Filter out invalid predictions for all models
    valid_idx = np.ones(len(ground_truth), dtype=bool)  # Start with all indices as valid
    for name in model_names:
        valid_idx &= preds_dict[name] != "invalid"  # Combine masks for all models

    # Apply the combined valid index mask to ground_truth and all models
    ground_truth = ground_truth[valid_idx]
    for name in model_names:
        preds_dict[name] = preds_dict[name][valid_idx]
    
    n = len(ground_truth)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        gt_sample = ground_truth[idx]
        for name in model_names:
            preds_sample = preds_dict[name][idx]
            rho, _ = spearmanr(preds_sample, gt_sample)
            rhos[name].append(rho)

    # Convert lists to arrays
    for name in rhos:
        rhos[name] = np.array(rhos[name])

    # Compute pairwise differences, CI, and p-values
    mean_diffs = {}
    ci = {}
    p_values = {}

    for model_i, model_j in itertools.combinations(model_names, 2):
        diffs = rhos[model_i] - rhos[model_j]
        mean_diff = np.mean(diffs)
        ci_lower = np.percentile(diffs, 2.5)
        ci_upper = np.percentile(diffs, 97.5)
        if mean_diff > 0:
            p_val = 2 * np.mean(diffs <= 0)
        else:
            p_val = 2 * np.mean(diffs >= 0)
        mean_diffs[(model_i, model_j)] = mean_diff
        ci[(model_i, model_j)] = (ci_lower, ci_upper)
        p_values[(model_i, model_j)] = p_val

    return {
        'mean_diffs': mean_diffs,
        'ci': ci,
        'p_values': p_values
    }


if __name__ == "__main__":
    # preds_dict = {
    #     'model_a': np.array([...]),
    #     'model_b': np.array([...]),
    #     'model_c': np.array([...])
    # }

    # ground_truth = np.array([...])
    n_bootstrap=50000
    input_dir = Path("/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/experiments/multilinguality/experiments")
    english_annotation_files = [
        file for file in input_dir.rglob("annotations*")
        if file.parent.name == "en"
    ]
    gt_file = Path("/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/prompt_based_annotations/test_data/gt/annotations__educational_content__en__gt.jsonl")
    aggregation_strategy = "majority"
    valid_labels = [0, 1, 2, 3, 4, 5]
    document_scores_df = get_document_scores_df(
        input_file_paths=english_annotation_files + [gt_file],
        aggregation_strategy=aggregation_strategy,
        valid_labels=valid_labels,
    )
    
    # convert df to dict
    preds_dict = {}
    for annotator in document_scores_df["annotator"].unique():
        preds_dict[annotator] = document_scores_df[
            document_scores_df["annotator"] == annotator
        ]["score"].values
        
    ground_truth = preds_dict.pop("gt")
    results = bootstrap_spearman_multi(
        preds_dict=preds_dict,
        ground_truth=ground_truth,
        n_bootstrap=n_bootstrap,
        seed=42,
    )

    # Print pairwise results
    for pair, diff in results['mean_diffs'].items():
        ci = results['ci'][pair]
        p = results['p_values'][pair]
        print(f"{pair[0]} vs {pair[1]}: Δρ = {diff:.4f}, 95% CI = ({ci[0]:.4f}, {ci[1]:.4f}), p = {p:.4f}")
