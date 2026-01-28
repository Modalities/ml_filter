from __future__ import annotations

import json
from pathlib import Path

from ml_filter.data_pipelines.filtering.paired_average_threshold_filter import PairedAverageThresholdFilter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_paired_average_threshold_filter_filters_by_average(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(
        text_dir / "a.jsonl",
        [
            {"id": "d1", "text": "keep"},
            {"id": "d2", "text": "drop"},
        ],
    )
    _write_jsonl(
        scores_dir / "a.jsonl",
        [
            {"id": "d1", "score_a": 0.5, "score_b": 1.5},
            {"id": "d2", "score_a": 0.2, "score_b": 0.4},
        ],
    )

    reader = PairedAverageThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_a", "score_b"],
        average_threshold=1.0,
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=False,
        glob_pattern=None,
    )

    docs = list(reader.read_file("a.jsonl"))
    assert [d.id for d in docs] == ["d1"]
    assert docs[0].text == "keep"


def test_paired_average_threshold_filter_per_folder_thresholds(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"

    _write_jsonl(
        text_dir / "Deu_Latn" / "a.jsonl",
        [
            {"id": "d1", "text": "drop"},
            {"id": "d2", "text": "keep"},
        ],
    )
    _write_jsonl(
        scores_dir / "Deu_Latn" / "a.jsonl",
        [
            {"id": "d1", "score_a": 1.4, "score_b": 1.4},
            {"id": "d2", "score_a": 1.6, "score_b": 1.6},
        ],
    )

    reader = PairedAverageThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score_a", "score_b"],
        average_threshold=0.0,
        average_thresholds_by_folder={"Deu_Latn": 1.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        recursive=True,
        glob_pattern=None,
    )

    docs = list(reader.read_file("Deu_Latn/a.jsonl"))
    assert [d.id for d in docs] == ["d2"]
