import pytest
from unittest.mock import MagicMock
import matplotlib.pyplot as plt

from ml_filter.analysis.plot_score_distributions import plot_scores, plot_differences_in_scores


@pytest.fixture
def mock_document_scores():
    return {
        "prompt1": {
            "doc1": {"version1": 3, "version2": 4},
            "doc2": {"version1": 5, "version2": 6},
            "doc3": {"version1": 7, "version2": 8},
        }
    }


def test_plot_scores(monkeypatch, tmp_path, mock_document_scores):   
    def mock_get_document_scores(*args, **kwargs):
        return mock_document_scores 
    monkeypatch.setattr("ml_filter.analysis.plot_score_distributions.get_document_scores", mock_get_document_scores)

    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    
    plot_scores(path_to_files=[], output_dir=output_dir, aggregation="mean")
    
    # Check that the expected file is created
    plot_path = output_dir / "prompt1_score_distributions.png"
    assert plot_path.exists(), "Plot file was not created."

    # Open and inspect the plot to ensure it is not empty
    plt.imread(plot_path)


def test_plot_differences_in_scores(monkeypatch, tmp_path, mock_document_scores):
    def mock_get_document_scores(*args, **kwargs):
        return mock_document_scores 
    monkeypatch.setattr("ml_filter.analysis.plot_score_distributions.get_document_scores", mock_get_document_scores)

    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    
    plot_differences_in_scores(path_to_files=[], output_dir=output_dir, aggregation="mean")
    
    # Check that the expected files are created
    histogram_path = output_dir / "prompt1_score_distributions_difference_histogram.png"
    boxplot_path = output_dir / "prompt1_score_distributions_difference_boxplot.png"
    
    assert histogram_path.exists(), "Histogram plot file was not created."
    assert boxplot_path.exists(), "Boxplot file was not created."

    # Open and inspect the plots to ensure they are not empty
    plt.imread(histogram_path)
    plt.imread(boxplot_path)
