import json

import pytest
import pandas as pd

from ml_filter.analysis.collect_ir_metrics import (
    style_df,
    read_metric_data,
    get_top_n_models,
    write_latex_output,
    aggregate_across_languages,
    plot_confusion_matrix,
    collect_ir_metrics
)


@pytest.fixture
def example_df():
    data = {
        "Model": ["model1", "model2", "model3"],
        "metric1": [0.8, 0.9, 0.85],
        "metric2": [0.2, 0.1, 0.15],
        "Invalid": [2, 3, 4],
        "lang": ["en", "en", "en"],
        "Filepath": ["path1", "path2", "path3"],
        "CM": [{"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}},
               {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}},
               {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}}]
    }
    return pd.DataFrame(data)


@pytest.fixture
def example_aggregated_metrics_df():
    data = {
        "metric1": {
            "model1": 0.8,
            "model2": 0.4,
            "model3": 0.5,
        },
        "metric2": {
            "model1": 0.2,
            "model2": 0.3,
            "model3": 0.5,
        }
    }
    return pd.DataFrame(data)


@pytest.fixture
def example_top_n_models():
    return {
        1: {
            "metric1": {"model1": 1, "model2": 1, "model3": 1},
            "metric2": {"model1": 1, "model2": 1, "model3": 1}
        }
    }


def test_style_df(example_df):
    styled_df = style_df(
        df=example_df,
        sort_key="Model",
        max_columns=["metric1"],
        min_columns=["metric2"]
    )
    assert isinstance(styled_df, pd.io.formats.style.Styler)


def test_read_metric_data(tmp_path):
    # Create example JSON files
    example_data = {
        "metrics": {"metric1": 0.8, "metric2": 0.2},
        "CM": {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}}
    }
    file_path = tmp_path / "ir_model1_model2.json"
    with file_path.open("w") as f:
        json.dump(example_data, f)
    
    df, metrics = read_metric_data(tmp_path)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(metrics, list)
    assert "Model" in df.columns
    assert "metric1" in metrics
    assert "metric2" in metrics


def test_get_top_n_models(example_df):
    top_n_models = get_top_n_models(
        df=example_df,
        top_n=1,
        max_metrics=["metric1"],
        min_metrics=["metric2"]
    )
    assert isinstance(top_n_models, dict)
    assert 1 in top_n_models
    assert "metric1" in top_n_models[1]
    assert "metric2" in top_n_models[1]


def test_write_latex_output(tmp_path, example_df, example_aggregated_metrics_df, example_top_n_models):
    output_directory = tmp_path
    write_latex_output(
        df=example_df,
        aggregated_metrics_df=example_aggregated_metrics_df,
        top_n_models=example_top_n_models,
        output_directory=output_directory,
        max_metrics=["metric1"],
        min_metrics=["metric2"]
    )
    output_file = output_directory / "ir_summary_gt.tex"
    assert output_file.exists()


def test_aggregate_across_languages(example_df):
    metrics = ["metric1", "metric2", "Invalid"]
    aggregated_metrics_df, aggregated_cm = aggregate_across_languages(
        df=example_df,
        metrics=metrics
    )
    assert isinstance(aggregated_metrics_df, pd.DataFrame)
    assert isinstance(aggregated_cm, dict)
    assert "metric1" in aggregated_metrics_df.columns
    assert "metric2" in aggregated_metrics_df.columns
    assert "Invalid" in aggregated_metrics_df.columns


def test_plot_confusion_matrix(tmp_path):
    aggregated_cm = {
        "model1": {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}},
        "model2": {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}}
    }
    output_directory = tmp_path
    plot_confusion_matrix(
        aggregated_cm=aggregated_cm,
        output_directory=output_directory
    )
    output_file = output_directory / "confusion_matrix_across_languages_model1.png"
    assert output_file.exists()


def test_collect_ir_metrics(tmp_path):
    input_directory = tmp_path / "input"
    input_directory.mkdir()
    output_directory = tmp_path / "output"
    output_directory.mkdir()

    # Create example JSON files
    example_data = {
        "metrics": {"metric1": 0.8, "metric2": 0.2, "Invalid": 2},
        "CM": {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}}
    }
    file_path = input_directory / "ir_model1_model2.json"
    with file_path.open("w") as f:
        json.dump(example_data, f)

    collect_ir_metrics(
        input_directory=input_directory,
        output_directory=output_directory,
        top_n=1,
        min_metrics=["metric2"]
    )
    output_file = output_directory / "ir_summary_gt.tex"
    assert output_file.exists()
