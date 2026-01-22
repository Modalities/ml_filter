from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_filter.data_processing.jsonl_filtering.paired_threshold_filter import PairedThresholdFilter


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_paired_threshold_domain_filter_filters(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"
    domains_dir = tmp_path / "domains"

    _write_jsonl(
        text_dir / "a.jsonl",
        [
            {"id": "d1", "text": "hello world"},
            {"id": "d2", "text": "filtered by score"},
            {"id": "d3", "text": "filtered by domain"},
        ],
    )
    _write_jsonl(
        scores_dir / "a.jsonl",
        [
            {"id": "d1", "score": 1.0},
            {"id": "d2", "score": 0.1},
            {"id": "d3", "score": 1.0},
        ],
    )
    _write_jsonl(
        domains_dir / "a.jsonl",
        [
            {"id": "d1", "domain": "wikipedia.org"},
            {"id": "d2", "domain": "wikipedia.org"},
            {"id": "d3", "domain": "spam.example"},
        ],
    )

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score"],
        thresholds_by_score_key={"score": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        domains_data_folder=str(domains_dir),
        accepted_domains=["wikipedia.org"],
        domain_jsonl_id_key="id",
        domain_jsonl_domain_key="domain",
        recursive=False,
        glob_pattern=None,
    )

    docs = list(reader.read_file("a.jsonl"))
    assert [d.id for d in docs] == ["d1"]
    assert docs[0].text == "hello world"


def test_paired_threshold_domain_filter_raises_when_domains_file_missing(tmp_path: Path):
    text_dir = tmp_path / "text"
    scores_dir = tmp_path / "scores"
    domains_dir = tmp_path / "domains"

    _write_jsonl(text_dir / "a.jsonl", [{"id": "d1", "text": "hello"}])
    _write_jsonl(scores_dir / "a.jsonl", [{"id": "d1", "score": 1.0}])

    reader = PairedThresholdFilter(
        text_data_folder=str(text_dir),
        scores_data_folder=str(scores_dir),
        score_keys=["score"],
        thresholds_by_score_key={"score": 0.5},
        text_jsonl_id_key="id",
        score_jsonl_id_key="id",
        text_jsonl_text_key="text",
        domains_data_folder=str(domains_dir),
        accepted_domains=["wikipedia.org"],
        domain_jsonl_id_key="id",
        domain_jsonl_domain_key="domain",
        recursive=False,
        glob_pattern=None,
    )

    with pytest.raises(FileNotFoundError, match="Paired domains JSONL file not found"):
        list(reader.read_file("a.jsonl"))
