
from collections import Counter
import json
from pathlib import Path
import statistics
from typing import List, Union

import krippendorff
import numpy as np
from scipy.stats import spearmanr, kendalltau
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa

from ml_filter.analysis.utils import get_document_scores


# Prepare data for Fleiss' Kappa
def prepare_fleiss_data(scores_list: List[list]):
    max_score = max(max(scores) for scores in scores_list)
    fleiss_data = np.zeros((len(scores_list), max_score + 1))
    for i, scores in enumerate(scores_list):
        for score in scores:
            fleiss_data[i, score] += 1
    return fleiss_data


# Compute pairwise correlations
def compute_pairwise_correlations(all_scores: List[list], metric: str):
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
    return np.mean(results)  # Average of all pairwise correlations


# Compute Krippendorff's Alpha
def compute_krippendorffs_alpha(all_scores: List[list]):
    flattened_scores = np.array(all_scores).T  # Transpose for Krippendorff's input
    return krippendorff.alpha(reliability_data=flattened_scores, level_of_measurement='ordinal')


def compute_doc_level_variation(all_scores: List[list], all_document_ids: List[str]):
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


# Main function to compute metrics
def compute_interrater_reliability_metrics(
    path_to_files: List[Path],
    output_file_path: Path,    
    single_annotator: bool = False,
    aggregation: Union[None, str] = None
) -> None:
    # check parameters
    if single_annotator and aggregation is not None:
        raise ValueError("aggregation types other than None are only valid when comparing multiple annotators")
    if not single_annotator and aggregation is None:
        raise ValueError("aggregation type must not be None when comparing multiple annotators")
    
    document_scores = get_document_scores(path_to_files, aggregation=aggregation)
    metrics = {}
    for prompt in document_scores:
        all_document_ids = []
        all_scores = []
        
        num_versions = 1 if single_annotator else max(len(versions) for versions in document_scores[prompt].values())
        for document_id, scores_per_version in document_scores[prompt].items():
            if single_annotator:
                if len(scores_per_version) != 1:
                    raise ValueError(f"There should be only one annotator if single_annotator is set to true, but found multiple for document id {document_id}: {list(scores_per_version.keys())}")
                all_scores.append(next(iter(scores_per_version.values())))
            else:
                scores = []
                for version in scores_per_version:
                    scores.append(scores_per_version[version])

                # skip documents where not a score for each version exists
                if len(scores) != num_versions:
                    continue
                
                all_scores.append(scores)
            all_document_ids.append(document_id)

        # some metrics accept only categories, not continuous values
        all_scores_rounded = [[round(val) for val in scores] for scores in all_scores]
        
        # Fleiss' Kappa
        fleiss_data = prepare_fleiss_data(all_scores_rounded)
        fk = fleiss_kappa(fleiss_data, method='fleiss')

        # Spearman's Rank Correlation (average of all pairwise)
        spearman_corr = compute_pairwise_correlations(all_scores, metric='spearman')

        # Kendall's Tau (average of all pairwise)
        kendall_corr = compute_pairwise_correlations(all_scores, metric='kendall')
        
        # Cohen's Kappa (average of all pairwise)
        cohen_kappa = compute_pairwise_correlations(all_scores_rounded, metric='cohen')

        # Krippendorff's Alpha
        kripp_alpha = compute_krippendorffs_alpha(all_scores)
        
        # variation per document
        doc_vars = compute_doc_level_variation(all_scores=all_scores_rounded, all_document_ids=all_document_ids)

        metrics[prompt] = {
            'Fleiss Kappa': fk,
            'Cohen Kappa (avg pairwise)': cohen_kappa,
            'Spearman Rank Correlation (avg pairwise)': spearman_corr,
            'Kendall Tau (avg pairwise)': kendall_corr,
            'Krippendorff Alpha': kripp_alpha,
            "Variation per Document": doc_vars
        }

    # print results and write them to the output file
    print("\n".join(f"{key}: {value}" for key, value in metrics.items()))
    
    with output_file_path.open("w") as f:
        json.dump(metrics, f, indent=4)
