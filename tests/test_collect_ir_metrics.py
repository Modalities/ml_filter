import json

import pandas as pd

from ml_filter.analysis.collect_ir_metrics import (
    style_df,
    read_metric_data,
    get_top_n_annotators,
    write_latex_output,
    aggregate_across_languages,
    plot_confusion_matrix,
    collect_ir_metrics
)


def test_style_df(example_df):
    styled_df = style_df(
        df=example_df,
        sort_key="Annotator",
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
    test_dir = tmp_path / "en"
    test_dir.mkdir()
    file_path = test_dir / "ir_annotator1_gt.json"
    with file_path.open("w") as f:
        json.dump(example_data, f)
    
    df, metrics = read_metric_data(tmp_path)
    expected_df = pd.DataFrame.from_dict({
        'metric1': {0: 0.8},
        'metric2': {0: 0.2},
        'Annotator': {0: 'annotator1'},
        'Filepath': {0: file_path},
        'CM': {0: {'0': {'0': 1, '1': 0}, '1': {'0': 0, '1': 1}}},
        'lang': {0: 'en'}
    })
    pd.testing.assert_frame_equal(df, expected_df)
    assert isinstance(metrics, list)
    assert "Annotator" in df.columns
    assert "metric1" in metrics
    assert "metric2" in metrics


def test_get_top_n_annotators(example_df):
    top_n_annotators = get_top_n_annotators(
        df=example_df,
        top_n=1,
        max_metrics=["metric1"],
        min_metrics=["metric2"]
    )
    assert isinstance(top_n_annotators, dict)
    assert 1 in top_n_annotators
    assert "metric1" in top_n_annotators[1]
    assert "metric2" in top_n_annotators[1]


def test_write_latex_output(tmp_path, example_df, example_aggregated_metrics_df, example_top_n_annotators):
    output_directory = tmp_path
    write_latex_output(
        df=example_df,
        aggregated_metrics_df=example_aggregated_metrics_df,
        top_n_annotators=example_top_n_annotators,
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
    expected_aggregated_metrics_df = pd.DataFrame.from_dict({
        'metric1': {'annotator1': 0.8, 'annotator2': 0.9, 'annotator3': 0.85},
        'metric2': {'annotator1': 0.2, 'annotator2': 0.1, 'annotator3': 0.15},
        'Invalid': {'annotator1': 2, 'annotator2': 3, 'annotator3': 4}
    })
    pd.testing.assert_frame_equal(aggregated_metrics_df, expected_aggregated_metrics_df)
    expected_aggregated_cm = {
        "annotator1": {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}},
        "annotator2": {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}},
        "annotator3": {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}}
    }
    assert aggregated_cm == expected_aggregated_cm


def test_plot_confusion_matrix(tmp_path):
    aggregated_cm = {
        "annotator1": {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}},
        "annotator2": {"0": {"0": 1, "1": 0}, "1": {"0": 0, "1": 1}}
    }
    output_directory = tmp_path
    plot_confusion_matrix(
        aggregated_cm=aggregated_cm,
        output_directory=output_directory
    )
    output_file = output_directory / "confusion_matrix_across_languages_annotator1.png"
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
    file_path = input_directory / "ir_annotator1_gt.json"
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
