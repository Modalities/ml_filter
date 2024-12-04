
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pytest
from statistics import mean, stdev


# Import functions to be tested
from ml_filter.analysis.interrater_reliability import (
    prepare_fleiss_data,
    compute_pairwise_correlations,
    compute_krippendorffs_alpha,
    compute_doc_level_variation,
    compute_interrater_reliability_metrics,
)

# Mock for `get_document_scores`
from ml_filter.analysis.utils import get_document_scores


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
    assert isinstance(spearman_corr, float), "Spearman correlation should return a float."
    assert isinstance(kendall_corr, float), "Kendall correlation should return a float."
    assert isinstance(cohen_corr, float), "Cohen's kappa should return a float."


def test_compute_krippendorffs_alpha(example_scores):
    result = compute_krippendorffs_alpha(example_scores)
    assert isinstance(result, float), "Krippendorff's alpha should return a float."


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


@pytest.mark.parametrize(
    "mock_document_scores, single_annotator, aggregation",
    [
        # Test case 1: Single annotator
        (
            {
                "prompt1": {
                    "doc1": {"version1": [3, 4, 5]},
                    "doc2": {"version1": [2, 4, 4]},
                    "doc3": {"version1": [3, 3, 3]},
                }
            },
            True,
            None
        ),
        # Test case 2: Multiple annotators
        (
            {
                "prompt1": {
                    "doc1": {"version1": 1, "version2": 1},
                    "doc2": {"version1": 2, "version2": 1},
                    "doc3": {"version1": 4, "version2": 5},
                }
            },
            False,
            "mean"
        ),
    ],
)
def test_compute_interrater_reliability_metrics_single_annotator(monkeypatch, tmp_path, mock_document_scores, single_annotator, aggregation):
    # Mock `get_document_scores` to return a controlled structure
    def mock_get_document_scores(*args, **kwargs):
        return mock_document_scores

    monkeypatch.setattr("ml_filter.analysis.interrater_reliability.get_document_scores", mock_get_document_scores)

    output_file = tmp_path / "output.json"

    # Call function
    compute_interrater_reliability_metrics(
        path_to_files=[],
        output_file_path=output_file,
        single_annotator=single_annotator,
        aggregation=aggregation,
    )

    # Verify output file exists
    assert output_file.exists(), "Output file was not created."

    # Verify content
    with output_file.open() as f:
        result = json.load(f)
        
    assert "prompt1" in result, "Output metrics should include the prompt."
    assert len(result["prompt1"]) > 0


def test_invalid_parameters():
    with pytest.raises(ValueError, match="aggregation type must not be None"):
        compute_interrater_reliability_metrics(
            path_to_files=[],
            output_file_path=Path("output.json"),
            single_annotator=False,
            aggregation=None,
        )

    with pytest.raises(ValueError, match="aggregation types other than None are only valid"):
        compute_interrater_reliability_metrics(
            path_to_files=[],
            output_file_path=Path("output.json"),
            single_annotator=True,
            aggregation="mean",
        )
