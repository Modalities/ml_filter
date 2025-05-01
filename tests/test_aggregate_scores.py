import json
from pathlib import Path
from unittest.mock import mock_open, patch

import pytest

from ml_filter.analysis.aggregate_scores import (
    _extract_annotator_name,
    aggregate_scores,
    remove_field_from_jsonl_file,
    write_scores_to_file,
)


def test_extract_annotator_name():
    # Arrange
    filename = Path("annotations_annotator1.jsonl")

    # Act
    annotator_name = _extract_annotator_name(filename)

    # Assert
    assert annotator_name == "annotator1"


@pytest.fixture
def raw_data_file_path(tmp_path):
    raw_data_file_path = tmp_path / "raw_data.jsonl"
    raw_data = [
        {"id": "doc1"},
        {"id": "doc2"},
    ]

    with raw_data_file_path.open("w") as f:
        for entry in raw_data:
            f.write(json.dumps(entry) + "\n")
    return str(raw_data_file_path)


@pytest.fixture
def create_annotation_file(tmp_path, raw_data_file_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()

    file = input_dir / "annotations__educational_prompt__en__test_annotator.jsonl"

    # Two annotations for the same document
    annotations = [
        {
            "document_id": "doc1",
            "scores": [1.0, 2.0, 3.0],
            "meta_information": {
                "raw_data_file_path": raw_data_file_path,
            },
        },
        {
            "document_id": "doc2",
            "scores": [2.0, 2.0, 1.0],
            "meta_information": {
                "raw_data_file_path": raw_data_file_path,
            },
        },
    ]

    with file.open("w") as f:
        for a in annotations:
            f.write(json.dumps(a) + "\n")
    return input_dir, file


def test_aggregate_scores(create_annotation_file, tmp_path):
    input_dir, _ = create_annotation_file
    output_dir = tmp_path / "output"

    aggregate_scores(
        input_directory=input_dir,
        output_directory=output_dir,
        aggregation_strategy="majority",
        valid_labels=list(range(6)),
        batch_size=2,
        raw_data_lookup_dir=tmp_path,
    )

    # Open the output file and check the contents
    output_file = output_dir / "raw_data_annotator_aggregated_scores_majority.jsonl"

    with output_file.open("r") as f:
        lines = f.readlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["id"] == "doc1"
        assert json.loads(lines[0])["score"] == 6.0 / 3
        assert json.loads(lines[1])["id"] == "doc2"
        assert json.loads(lines[1])["score"] == 2.0


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
