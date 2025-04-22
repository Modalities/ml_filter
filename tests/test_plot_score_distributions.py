
import matplotlib.pyplot as plt
from pathlib import Path
import pytest

from ml_filter.analysis.plot_score_distributions import plot_scores, plot_differences_in_scores


@pytest.fixture
def path_to_files():
    paths = [
        Path("tests/resources/data/llm_annotations/en/annotations_edu_en_test_1.jsonl"),
        Path("tests/resources/data/llm_annotations/en/annotations_edu_en_test_2.jsonl"),
        Path("tests/resources/data/llm_annotations/en/annotations_edu_en_gt_1.jsonl"),
    ]
    return paths


def test_plot_scores(tmp_path, path_to_files):   
    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    
    plot_scores(path_to_files=path_to_files, output_dir=output_dir, aggregation="majority", labels=[0, 1, 2, 3, 4, 5])
    
    # Check that the expected plot was created
    plot_path = output_dir / f"edu_score_distributions_test.png"
    assert plot_path.exists(), "Plot file was not created."

    # Open and inspect the plot to ensure it is not empty
    plt.imread(plot_path)


def test_plot_differences_in_scores(tmp_path, path_to_files):
    output_dir = tmp_path / "plots"
    output_dir.mkdir()
    aggregation = "majority"
    plot_differences_in_scores(path_to_files=path_to_files, output_dir=output_dir, aggregation=aggregation, labels=[0, 1, 2, 3, 4, 5])
    
    # Check that the expected files were created
    histogram_path = output_dir / f"edu_score_distributions_difference_histogram_{aggregation}.png"
    boxplot_path = output_dir / f"edu_score_distributions_difference_boxplot_{aggregation}.png"
    
    assert histogram_path.exists(), "Histogram plot file was not created."
    assert boxplot_path.exists(), "Boxplot file was not created."

    # Open and inspect the plots to ensure they are not empty
    plt.imread(histogram_path)
    plt.imread(boxplot_path)
