from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import itertools

from ml_filter.analysis.utils import get_document_scores_df


def permutation_spearman_multi(preds_dict, ground_truth, n_permutations=10000, seed=42):
    """
    Computes permutation test estimates of pairwise differences in Spearman correlation
    between multiple models and the ground truth.

    Args:
        preds_dict: dict of model_name -> prediction array (length N)
        ground_truth: array of length N
        n_permutations: number of permutations
        seed: random seed

    Returns:
        A dict of results:
            - mean_diffs: dict of (model_i, model_j) -> observed mean difference
            - p_values: dict of (model_i, model_j) -> two-sided p-value
    """
    rng = np.random.default_rng(seed)
    model_names = list(preds_dict.keys())

    # Filter out invalid predictions for all models
    valid_idx = np.ones(len(ground_truth), dtype=bool)  # Start with all indices as valid
    for name in model_names:
        valid_idx &= preds_dict[name] != "invalid"  # Combine masks for all models

    # Apply the combined valid index mask to ground_truth and all models
    ground_truth = ground_truth[valid_idx]
    for name in model_names:
        preds_dict[name] = preds_dict[name][valid_idx]
        
    # Compute observed Spearman correlations
    observed_rhos = {}
    for name in model_names:
        rho, _ = spearmanr(preds_dict[name], ground_truth)
        observed_rhos[name] = rho

    # Compute observed pairwise differences
    mean_diffs = {}
    for model_i, model_j in itertools.combinations(model_names, 2):
        mean_diffs[(model_i, model_j)] = observed_rhos[model_i] - observed_rhos[model_j]

    # Perform permutation test
    p_values = {}
    for model_i, model_j in itertools.combinations(model_names, 2):
        diffs = []
        for _ in range(n_permutations):
            # Shuffle the ground truth
            permuted_ground_truth = rng.permutation(ground_truth)

            # Compute Spearman correlations for the permuted data
            rho_i, _ = spearmanr(preds_dict[model_i], permuted_ground_truth)
            rho_j, _ = spearmanr(preds_dict[model_j], permuted_ground_truth)

            # Compute the difference in Spearman correlations
            diffs.append(rho_i - rho_j)

        # Convert diffs to a NumPy array
        diffs = np.array(diffs)

        # Calculate the p-value (two-sided)
        observed_diff = mean_diffs[(model_i, model_j)]
        p_val = np.mean(np.abs(diffs) >= np.abs(observed_diff))
        p_values[(model_i, model_j)] = p_val

    return {
        'mean_diffs': mean_diffs,
        'p_values': p_values
    }


if __name__ == "__main__":
    # preds_dict = {
    #     'model_a': np.array([...]),
    #     'model_b': np.array([...]),
    #     'model_c': np.array([...])
    # }

    # ground_truth = np.array([...])
    n_permutations=1000
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
    results = permutation_spearman_multi(
        preds_dict=preds_dict,
        ground_truth=ground_truth,
        n_permutations=n_permutations,
        seed=42,
    )

    # Print pairwise results
    for pair, diff in results['mean_diffs'].items():
        p = results['p_values'][pair]
        print(f"{pair[0]} vs {pair[1]}: Δρ = {diff:.4f}, p = {p:.4f}")
