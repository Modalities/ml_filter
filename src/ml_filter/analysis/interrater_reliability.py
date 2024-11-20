
import json
import numpy as np
from statsmodels.stats.inter_rater import fleiss_kappa
from sklearn.metrics import cohen_kappa_score


# Load JSONL file
def load_jsonl(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# Prepare data for Fleiss' Kappa
def prepare_fleiss_data(scores_list):
    # Convert the scores to frequency counts per category
    # Example: If scores = [[1, 2, 3], [2, 2, 3]], transform to [[0, 1, 2], [0, 2, 1]]
    max_score = max(max(scores) for scores in scores_list)
    fleiss_data = np.zeros((len(scores_list), max_score + 1))
    for i, scores in enumerate(scores_list):
        for score in scores:
            fleiss_data[i, score] += 1
    return fleiss_data


# Main function to compute metrics
def compute_metrics(jsonl_path):
    data = load_jsonl(jsonl_path)
    all_scores = [[int(score) for score in item['scores']] for item in data]

    # Fleiss' Kappa
    fleiss_data = prepare_fleiss_data(all_scores)
    fk = fleiss_kappa(fleiss_data, method='fleiss')

    # Cohen's Kappa (pairwise, as an example for Group 1 and Group 2)
    ck = cohen_kappa_score(
        [scores[0] for scores in all_scores],
        [scores[1] for scores in all_scores]
    )

    return {
        'Fleiss Kappa': fk,
        'Cohen Kappa (Group 1 & 2)': ck
    }

# Example Usage
jsonl_file_path = '/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/human_annotations_eurolingua.jsonl'
results = compute_metrics(jsonl_file_path)
print(results)
