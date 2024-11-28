
from ml_filter.analysis.interrater_reliability import compute_interrater_reliability_metrics


def test_compute_interrater_reliability_metrics():
    """Test the successful execution of compute_interrater_reliability_metrics without errors."""
    compute_interrater_reliability_metrics("tests/resources/data/annotations.jsonl")
 