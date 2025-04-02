import json
from pathlib import Path
from unittest.mock import MagicMock, patch, mock_open
from ml_filter.analysis.aggregate_scores import (
    _extract_annotator_name,
    aggregate_scores_in_directory,
    write_scores_to_file,
    aggregate_human_annotations,
    remove_field_from_jsonl_file,
)


def test_extract_annotator_name():
    # Arrange
    filename = Path("annotations_annotator1.jsonl")
    
    # Act
    annotator_name = _extract_annotator_name(filename)
    
    # Assert
    assert annotator_name == "annotator1"


@patch("ml_filter.analysis.aggregate_scores.get_document_scores")
@patch("ml_filter.analysis.aggregate_scores.add_scores")
def test_aggregate_scores_in_directory(mock_add_scores, mock_get_document_scores, tmp_path):
    # Arrange
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    (input_directory / "annotations_annotator1.jsonl").write_text("[]")
    
    # Mock the return value of get_document_scores to simulate a DataFrame
    mock_document_scores_df = MagicMock()
    mock_document_scores_df["raw_data_file_path"].unique.return_value = ["file1.jsonl"]
    mock_get_document_scores.return_value = mock_document_scores_df

    # Act
    aggregate_scores_in_directory(
        input_directory=input_directory,
        output_directory=output_directory,
        aggregation="majority",
        labels=[0, 1, 2, 3, 4, 5],
    )

    # Assert
    mock_add_scores.assert_called()


@patch("ml_filter.analysis.aggregate_scores.json.loads")
@patch("ml_filter.analysis.aggregate_scores.json.dumps")
@patch("ml_filter.analysis.aggregate_scores.Path.open", new_callable=mock_open)
def test_add_scores(mock_open_file, mock_json_dumps, mock_json_loads):
    # Arrange
    output_file_path = Path("output.jsonl")
    raw_data_file_path = Path("raw_data.jsonl")
    document_scores_for_raw_data_dict = {"doc1": 0.5}
    mock_json_loads.side_effect = [{"id": "doc1"}]
    mock_json_dumps.side_effect = lambda obj, ensure_ascii: json.dumps(obj)

    # Act
    write_scores_to_file(
        output_file_path=output_file_path,
        raw_data_file_path=raw_data_file_path,
        document_scores_for_raw_data_dict=document_scores_for_raw_data_dict,
        aggregation="majority",
        batch_size=1,
        id_field="id",
    )

    # Assert
    mock_open_file.assert_called()


@patch("ml_filter.analysis.aggregate_scores.get_document_scores")
@patch("ml_filter.analysis.aggregate_scores.add_scores")
def test_aggregate_scores_in_directory(mock_add_scores, mock_get_document_scores, tmp_path):
    # Arrange
    input_directory = tmp_path / "input"
    output_directory = tmp_path / "output"
    input_directory.mkdir()
    (input_directory / "annotations_annotator1.jsonl").write_text("[]")
    
    # Mock the return value of get_document_scores to simulate a DataFrame
    mock_document_scores_df = MagicMock()
    mock_document_scores_df["raw_data_file_path"].unique.return_value = ["file1.jsonl"]
    mock_get_document_scores.return_value = mock_document_scores_df

    # Act
    aggregate_scores_in_directory(
        input_directory=input_directory,
        output_directory=output_directory,
        aggregation="majority",
        labels=[0, 1, 2, 3, 4, 5],
    )

    # Assert
    mock_add_scores.assert_called()
    

@patch("ml_filter.analysis.aggregate_scores.json.loads")
def test_remove_field_from_jsonl_file(mock_json_loads, tmp_path):
    # Arrange
    jsonl_file_path = tmp_path / "file.jsonl"
    temp_file_path = tmp_path / "file.tmp"
    jsonl_file_path.write_text('{"key": "value", "scores": 0.5}\n')
    mock_json_loads.side_effect = [{"key": "value", "scores": 0.5}]

    # Act
    remove_field_from_jsonl_file(jsonl_file_path, "scores")

    # Assert
    assert not temp_file_path.exists()  # Ensure the temporary file is removed
    with jsonl_file_path.open("r", encoding="utf-8") as f:
        content = f.read()
        assert content == '{"key": "value"}'