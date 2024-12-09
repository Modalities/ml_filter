
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


@pytest.mark.parametrize(
    "path_to_files, single_annotator, aggregation",
    [
        # Test case 1: Single annotator
        (
            [
                "tests/resources/data/llm_annotations/en/annotations_edu_en_test_1.jsonl",
                "tests/resources/data/llm_annotations/en/annotations_edu_en_test_2.jsonl",
            ],    
            True,
            None
        ),
        # Test case 2: Multiple annotators
        (
            [
                "tests/resources/data/llm_annotations/en/annotations_edu_en_test_1.jsonl",
                "tests/resources/data/llm_annotations/en/annotations_edu_en_test_2.jsonl",
                "tests/resources/data/llm_annotations/de/annotations_edu_de_test_1.jsonl",
                "tests/resources/data/llm_annotations/de/annotations_edu_de_test_2.jsonl",
                "tests/resources/data/llm_annotations/en/annotations_toxic_en_test_1.jsonl",
                "tests/resources/data/llm_annotations/en/annotations_toxic_en_test_2.jsonl",
                "tests/resources/data/llm_annotations/de/annotations_toxic_de_test_1.jsonl",
            ],
            False,
            "mean"
        ),
    ],
)
def test_compute_interrater_reliability_metrics_single_annotator(tmp_path, path_to_files, single_annotator, aggregation):
    output_file = tmp_path / "output.json"
    path_to_files = [Path(p) for p in path_to_files]

    # Call function
    compute_interrater_reliability_metrics(
        path_to_files=path_to_files,
        output_file_path=output_file,
        single_annotator=single_annotator,
        aggregation=aggregation,
    )

    # Verify output file exists
    assert output_file.exists(), "Output file was not created."

    # Verify content
    with output_file.open() as f:
        result = json.load(f)
        
    assert "edu" in result, "Output metrics should include the prompt."
    assert len(result["edu"]) > 0


def test_invalid_parameters():
    with pytest.raises(ValueError):
        compute_interrater_reliability_metrics(
            path_to_files=[],
            output_file_path=Path("output.json"),
            single_annotator=False,
            aggregation=None,
        )

    with pytest.raises(ValueError):
        compute_interrater_reliability_metrics(
            path_to_files=[],
            output_file_path=Path("output.json"),
            single_annotator=True,
            aggregation="mean",
        )
