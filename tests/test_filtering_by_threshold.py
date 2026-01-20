import json
from pathlib import Path

import pytest

from ml_filter.data_processing.jsonl_filtering.filtering_by_threshold import ScoresParser


class _DummyDataFolder:
    def open(self, filepath: str, mode: str, compression=None):
        # BaseDiskReader passes relative paths sometimes; keep it simple.
        return open(filepath, mode, encoding="utf-8")


def _write_jsonl(tmp_path: Path, rows: list[dict]) -> str:
    p = tmp_path / "scores.jsonl"
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def test_filters_by_threshold_and_preserves_order(tmp_path: Path):
    fp = _write_jsonl(
        tmp_path,
        [
            {"document_id": "a", "tox": 0.1, "quality": 0.9},
            {"document_id": "b", "tox": 0.9, "quality": 0.9},  # fails tox
            {"document_id": "c", "tox": 0.0, "quality": 0.1},  # fails quality
            {"document_id": "d", "tox": 0.2, "quality": 0.95},
        ],
    )

    parser = ScoresParser(
        data_folder=_DummyDataFolder(),
        score_keys=["tox", "quality"],
        thresholds_by_score_key={"tox": 0.2, "quality": 0.9},
    )

    ids, scores = parser._parse_scores_jsonl_file(fp)
    assert ids == ["d"]
    assert scores == [{"tox": 0.2, "quality": 0.95}]


def test_duplicate_document_ids_are_disambiguated(tmp_path: Path):
    fp = _write_jsonl(
        tmp_path,
        [
            {"document_id": "x", "s": 1.0},
            {"document_id": "x", "s": 1.0},
            {"document_id": "x", "s": 1.0},
        ],
    )

    parser = ScoresParser(
        data_folder=_DummyDataFolder(),
        score_keys=["s"],
        thresholds_by_score_key={"s": 0.0},
    )

    ids, scores = parser._parse_scores_jsonl_file(fp)
    assert ids == ["x", "x_1", "x_2"]
    assert scores == [{"s": 1.0}, {"s": 1.0}, {"s": 1.0}]


def test_missing_thresholds_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        ScoresParser(
            data_folder=_DummyDataFolder(),
            score_keys=["a", "b"],
            thresholds_by_score_key={"a": 0.0},
        )
