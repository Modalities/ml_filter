
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from statistics import mean, stdev


# Import functions to be tested
from ml_filter.analysis.interrater_reliability import (
    compare_annotator_to_gt,
    compute_accuracy_mae_mse_against_gt,
    compute_metrics,
    prepare_fleiss_data,
    compute_pairwise_correlations,
    compute_krippendorffs_alpha,
    compute_doc_level_variation,
    compute_interrater_reliability_metrics,
)


@pytest.fixture
def example_scores():
    return [[3, 4, 5], [2, 4, 4], [3, 3, 3]]


@pytest.fixture
def example_ids():
    return ["doc1", "doc2", "doc3"]


def test_prepare_fleiss_data(example_scores):
    result = prepare_fleiss_data(example_scores)
    expected = np.array([[0, 0, 0, 1, 1, 1],
                         [0, 0, 1, 0, 2, 0],
                         [0, 0, 0, 3, 0, 0]])
    assert np.array_equal(result, expected), "Fleiss data not computed correctly."


def test_compute_pairwise_correlations(example_scores):
    spearman_corr = compute_pairwise_correlations(example_scores, metric="spearman")
    kendall_corr = compute_pairwise_correlations(example_scores, metric="kendall")
    cohen_corr = compute_pairwise_correlations(example_scores, metric="cohen")

    # Verify values
    assert spearman_corr == pytest.approx(0.1220084679281462, rel=1e-4), "Spearman correlation not computed correctly."
    assert kendall_corr == pytest.approx(0.10549886030924203, rel=1e-4), "Kendall correlation not computed correctly."
    assert cohen_corr == pytest.approx(0.2619047619047619, rel=1e-4), "Cohen's kappa not computed correctly."


def test_compute_krippendorffs_alpha(example_scores):
    krippendorffs_alpha = compute_krippendorffs_alpha(example_scores)
    assert krippendorffs_alpha == pytest.approx(0.0062893081761006275, rel=1e-4), "Krippendorff's alpha not computed correctly."


def test_compute_doc_level_variation(example_scores, example_ids):
    result = compute_doc_level_variation(example_scores, example_ids)

    # Verify structure
    assert "counts" in result, "Result should include 'counts'."
    assert "mean" in result, "Result should include 'mean'."
    assert "stdev" in result, "Result should include 'stdev'."

    # Verify values
    expected_counts = Counter([2, 2, 0])  # Variations for example_scores
    assert result["counts"] == expected_counts, "Counts are incorrect."
    assert result["mean"] == mean([2, 2, 0]), "Mean is incorrect."
    assert result["stdev"] == pytest.approx(stdev([2, 2, 0]), rel=1e-2), "Standard deviation is incorrect."


def test_compute_accuracy_mae_mse_against_gt():
    scores_0 = [1, 2, 3]
    scores_1 = [1, 3, 2]
    metrics = compute_accuracy_mae_mse_against_gt(scores_0, scores_1)
    expected_metrics = {'acc': 0.3333333333333333, 'mae': 0.6666666666666666, 'mse': 0.6666666666666666}
    assert metrics == expected_metrics


def test_compute_metrics():
    data = {
        "score_0": [1, 2, 3],
        "score_1": [1, 2, 3],
        "rounded_score_0": [1, 2, 3],
        "rounded_score_1": [1, 2, 3],
        "doc_id": [1, 2, 3]
    }
    thresholds = [2, 3]
    df = pd.DataFrame(data)
    metrics = compute_metrics(
        num_total_docs=3,
        valid_docs_df=df,
        thresholds=thresholds
    )
    expected_metrics = {
        'metrics': {
            'Fleiss': 1.0,
            'Cohen': 1.0,
            'Spearman': 1.0,
            'Kendall': 1.0,
            'Krippendorff': 1.0,
            'Invalid': 0,
            'TA-2': 1.0,
            'TA-3': 1.0,
            'CA_1': 1.0,
            'CA_2': 1.0,
            'CA_3': 1.0,
        },
        'Variation per Document': {1: 0, 2: 0, 3: 0, 'counts': {0: 3}, 'mean': 0, 'stdev': 0.0}
        }
    assert metrics == expected_metrics


def test_compare_model_to_gt(tmp_path):
    data = {
        "score_0": [1, 2, 3],
        "score_1": [1, 2, 3],
        "rounded_score_0": [1, 2, 3],
        "rounded_score_1": [1, 2, 3],
        "doc_id": [1, 2, 3]
    }
    df = pd.DataFrame(data)
    metrics = {"metrics": {}}
    output_dir = tmp_path
    updated_metrics = compare_annotator_to_gt(
        annotators=["gt", "model"],
        valid_docs_df=df,
        common_docs_df=df,
        metrics=metrics,
        output_dir=output_dir
    )
    expected_updated_metrics = {
        'metrics': {'Acc': 1.0, 'MAE': 0.0, 'MSE': 0.0},
        'CM': {
            0: {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            1: {-1: 0, 0: 0, 1: 1, 2: 0, 3: 0, 4: 0, 5: 0},
            2: {-1: 0, 0: 0, 1: 0, 2: 1, 3: 0, 4: 0, 5: 0},
            3: {-1: 0, 0: 0, 1: 0, 2: 0, 3: 1, 4: 0, 5: 0},
            4: {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            5: {-1: 0, 0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        }
    }
    assert updated_metrics == expected_updated_metrics


@pytest.mark.parametrize(
    "aggregation",
    [
        "majority",
        "mean",
        "median",
        "max",
        "min",
    ],
)
def test_compute_interrater_reliability_metrics(tmp_path, aggregation):
    path_to_files = [
        Path("tests/resources/data/llm_annotations/en/annotations__edu__en__test__1.jsonl"),
        Path("tests/resources/data/llm_annotations/en/annotations__edu__en__test__2.jsonl"),
        Path("tests/resources/data/llm_annotations/en/annotations__edu__en__gt__1.jsonl"),
    ]
    aggregation = "majority"
    output_dir = tmp_path / "interrater_reliability_metrics"
    labels = list(range(6))
    thresholds = [2, 3]

    # Call function
    compute_interrater_reliability_metrics(
        path_to_files=path_to_files,
        output_dir=output_dir,
        aggregation=aggregation,
        labels=labels,
        thresholds=thresholds,
    )

    # Verify output file exists
    assert output_dir.exists(), "Output file was not created."
    files = list(output_dir.iterdir())
    assert len(files) > 0, "Output directory should not be empty."
    
    for path in files:
        if path.suffix == ".json":
            # Verify content
            with path.open() as f:
                result = json.load(f)
        
            assert "metrics" in result, "No metrics found in output file."
            assert len(result["metrics"]) > 0, "No metrics found in output file."
