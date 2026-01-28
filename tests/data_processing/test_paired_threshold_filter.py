from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_filter.data_pipelines.filtering.paired_threshold_filter import PairedThresholdFilter




def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_paired_threshold_filter_happy_path_filters(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(
        text_dir / "a.jsonl",
        [
            {"id": "d1", "text": "hello world"},
            {"id": "d2", "text": "this should be filtered"},
        ],
    )
    _write_jsonl(
        scores_dir / "a.jsonl",
        [
            {"id": "d1", "score_Gemma_Snowflake": 1.0},
            {"id": "d2", "score_Gemma_Snowflake": 0.0},
        ],
    )

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_Gemma_Snowflake"],
        thresholds_by_score_key={"score_Gemma_Snowflake": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=False,
        glob_pattern=None,
    )

    docs = list(reader.read_file("a.jsonl"))
    assert [d.id for d in docs] == ["d1"]
    assert docs[0].text == "hello world"


def test_paired_threshold_filter_raises_on_id_mismatch(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(text_dir / "a.jsonl", [{"id": "d1", "text": "hello"}])
    _write_jsonl(scores_dir / "a.jsonl", [{"id": "d2", "score_Gemma_Snowflake": 1.0}])

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_Gemma_Snowflake"],
        thresholds_by_score_key={"score_Gemma_Snowflake": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=False,
        glob_pattern=None,
    )

    with pytest.raises(ValueError, match="mismatch"):
        list(reader.read_file("a.jsonl"))


def test_paired_threshold_filter_skip_line_on_id_mismatch(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(
        text_dir / "a.jsonl",
        [
            {"id": "d1", "text": "hello"},
            {"id": "dX", "text": "this line mismatches"},
            {"id": "d2", "text": "keep me"},
        ],
    )
    _write_jsonl(
        scores_dir / "a.jsonl",
        [
            {"id": "d1", "score_Gemma_Snowflake": 1.0},
            {"id": "d1", "score_Gemma_Snowflake": 1.0},
            {"id": "d2", "score_Gemma_Snowflake": 1.0},
        ],
    )

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_Gemma_Snowflake"],
        thresholds_by_score_key={"score_Gemma_Snowflake": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=False,
        glob_pattern=None,
        on_mismatch="skip_line",
    )

    docs = list(reader.read_file("a.jsonl"))
    assert [d.id for d in docs] == ["d1", "d2"]


def test_paired_threshold_filter_skip_file_on_id_mismatch(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(
        text_dir / "a.jsonl",
        [
            {"id": "d1", "text": "hello"},
            {"id": "dX", "text": "this line mismatches"},
            {"id": "d2", "text": "should not be reached"},
        ],
    )
    _write_jsonl(
        scores_dir / "a.jsonl",
        [
            {"id": "d1", "score_Gemma_Snowflake": 1.0},
            {"id": "d1", "score_Gemma_Snowflake": 1.0},
            {"id": "d2", "score_Gemma_Snowflake": 1.0},
        ],
    )

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_Gemma_Snowflake"],
        thresholds_by_score_key={"score_Gemma_Snowflake": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=False,
        glob_pattern=None,
        on_mismatch="skip_file",
    )

    docs = list(reader.read_file("a.jsonl"))
    assert [d.id for d in docs] == ["d1"]


def test_paired_threshold_filter_max_mismatches_per_file(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(
        text_dir / "a.jsonl",
        [
            {"id": "t1", "text": "a"},
            {"id": "t2", "text": "b"},
            {"id": "t3", "text": "c"},
        ],
    )
    _write_jsonl(
        scores_dir / "a.jsonl",
        [
            {"id": "s1", "score_Gemma_Snowflake": 1.0},
            {"id": "s2", "score_Gemma_Snowflake": 1.0},
            {"id": "s3", "score_Gemma_Snowflake": 1.0},
        ],
    )

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_Gemma_Snowflake"],
        thresholds_by_score_key={"score_Gemma_Snowflake": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=False,
        glob_pattern=None,
        on_mismatch="skip_line",
        max_mismatches_per_file=2,
    )

    with pytest.raises(ValueError, match="Too many id mismatches"):
        list(reader.read_file("a.jsonl"))


def test_paired_threshold_filter_raises_on_line_count_mismatch(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(
        text_dir / "a.jsonl",
        [
            {"id": "d1", "text": "hello"},
            {"id": "d2", "text": "extra"},
        ],
    )
    _write_jsonl(scores_dir / "a.jsonl", [{"id": "d1", "score_Gemma_Snowflake": 1.0}])

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_Gemma_Snowflake"],
        thresholds_by_score_key={"score_Gemma_Snowflake": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=False,
        glob_pattern=None,
    )

    with pytest.raises(ValueError, match="Line-count mismatch"):
        list(reader.read_file("a.jsonl"))


def test_paired_threshold_filter_raises_on_missing_score_keys(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(text_dir / "a.jsonl", [{"id": "d1", "text": "hello"}])
    _write_jsonl(scores_dir / "a.jsonl", [{"id": "d1", "other": 1.0}])

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_Gemma_Snowflake"],
        thresholds_by_score_key={"score_Gemma_Snowflake": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=False,
        glob_pattern=None,
    )

    with pytest.raises(ValueError, match="Missing score keys"):
        list(reader.read_file("a.jsonl"))


def test_paired_threshold_filter_raises_when_text_file_missing(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    # Only create the scores file.
    _write_jsonl(scores_dir / "shard01" / "a.jsonl", [{"id": "d1", "score_Gemma_Snowflake": 1.0}])

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_Gemma_Snowflake"],
        thresholds_by_score_key={"score_Gemma_Snowflake": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=True,
        glob_pattern=None,
    )

    with pytest.raises(FileNotFoundError, match="Paired text JSONL file not found"):
        list(reader.read_file("shard01/a.jsonl"))


def test_paired_threshold_filter_per_folder_thresholds(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(
        text_dir / "Deu_Latn" / "a.jsonl",
        [
            {"id": "d1", "text": "hello"},
            {"id": "d2", "text": "keep me"},
        ],
    )
    _write_jsonl(
        scores_dir / "Deu_Latn" / "a.jsonl",
        [
            {"id": "d1", "score_Gemma_Snowflake": 1.4},
            {"id": "d2", "score_Gemma_Snowflake": 1.6},
        ],
    )

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_Gemma_Snowflake"],
        thresholds_by_score_key={"score_Gemma_Snowflake": 0.5},
        thresholds_by_folder={
            "Deu_Latn": {"score_Gemma_Snowflake": 1.5},
        },
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=True,
        glob_pattern=None,
    )

    docs = list(reader.read_file("Deu_Latn/a.jsonl"))
    assert [d.id for d in docs] == ["d2"]
