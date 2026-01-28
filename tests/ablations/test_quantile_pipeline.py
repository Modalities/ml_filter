import json
from pathlib import Path

import yaml

from ml_filter.data_pipelines.quantile.quantile_steps import QuantileJsonlReader


def test_average_scores_and_threshold(tmp_path: Path) -> None:
    rows = [
        {"id": "1", "score_llama": 0, "score_mistral": 1, "score_gemma": 2},
        {"id": "2", "score_llama": 2, "score_mistral": 2, "score_gemma": 2},
        {"id": "3", "score_llama": 3, "score_mistral": 4, "score_gemma": 5},
        {"id": "4", "score_llama": 1, "score_mistral": 1, "score_gemma": 1},
    ]
    file_path = tmp_path / "lang_a" / "lang_a_part_1.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report_path = tmp_path / "quantile_report.yaml"
    reader = QuantileJsonlReader(
        data_folder=str(tmp_path),
        score_fields=["score_llama", "score_mistral", "score_gemma"],
        selection_quantile=0.5,
        report_path=report_path,
        glob_pattern="**/*.jsonl",
    )
    list(reader.run(rank=0, world_size=1))
    docs = list(yaml.safe_load_all(report_path.read_text(encoding="utf-8")))
    report = {doc["language"]: doc for doc in docs}
    lang_report = report["lang_a"]

    assert lang_report["selection_threshold"] == 1.5
    assert lang_report["averaged_score_counts"] == {1.0: 2, 2.0: 1, 4.0: 1}


def test_threshold_same_for_split_files(tmp_path: Path) -> None:
    rows = [
        {"id": "1", "score_llama": 0, "score_mistral": 0, "score_gemma": 0},
        {"id": "2", "score_llama": 2, "score_mistral": 2, "score_gemma": 2},
        {"id": "3", "score_llama": 4, "score_mistral": 4, "score_gemma": 4},
        {"id": "4", "score_llama": 6, "score_mistral": 6, "score_gemma": 6},
    ]
    single_path = tmp_path / "lang_single" / "lang_single_part_1.jsonl"
    single_path.parent.mkdir(parents=True, exist_ok=True)
    with single_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    split_paths = [
        tmp_path / "lang_split" / "lang_split_part_1.jsonl",
        tmp_path / "lang_split" / "lang_split_part_2.jsonl",
    ]
    split_paths[0].parent.mkdir(parents=True, exist_ok=True)
    with split_paths[0].open("w", encoding="utf-8") as f:
        for row in rows[:2]:
            f.write(json.dumps(row) + "\n")
    with split_paths[1].open("w", encoding="utf-8") as f:
        for row in rows[2:]:
            f.write(json.dumps(row) + "\n")

    report_path = tmp_path / "quantile_report.yaml"
    reader = QuantileJsonlReader(
        data_folder=str(tmp_path),
        score_fields=["score_llama", "score_mistral", "score_gemma"],
        selection_quantile=0.5,
        report_path=report_path,
        glob_pattern="**/*.jsonl",
    )
    list(reader.run(rank=0, world_size=1))
    docs = list(yaml.safe_load_all(report_path.read_text(encoding="utf-8")))
    report = {doc["language"]: doc for doc in docs}

    assert report["lang_single"]["selection_threshold"] == report["lang_split"]["selection_threshold"]
    assert report["lang_single"]["averaged_score_counts"] == report["lang_split"]["averaged_score_counts"]


def test_quantile_interpolation(tmp_path: Path) -> None:
    rows = [
        {"id": "1", "score_llama": 0, "score_mistral": 0, "score_gemma": 0},
        {"id": "2", "score_llama": 2, "score_mistral": 2, "score_gemma": 2},
        {"id": "3", "score_llama": 4, "score_mistral": 4, "score_gemma": 4},
        {"id": "4", "score_llama": 6, "score_mistral": 6, "score_gemma": 6},
    ]
    file_path = tmp_path / "lang_interp" / "lang_interp_part_1.jsonl"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    report_path = tmp_path / "quantile_report.yaml"
    reader = QuantileJsonlReader(
        data_folder=str(tmp_path),
        score_fields=["score_llama", "score_mistral", "score_gemma"],
        selection_quantile=0.25,
        report_path=report_path,
        glob_pattern="**/*.jsonl",
    )
    list(reader.run(rank=0, world_size=1))
    docs = list(yaml.safe_load_all(report_path.read_text(encoding="utf-8")))
    report = {doc["language"]: doc for doc in docs}

    assert report["lang_interp"]["selection_threshold"] == 1.5
