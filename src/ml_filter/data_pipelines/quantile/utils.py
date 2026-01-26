"""Shared helpers for ablation pipelines."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import yaml
from datatrove.io import DataFileLike, DataFolderLike, get_shard_from_paths_file


def parse_score(value) -> float:
    """Convert a score value to float or raise when invalid."""
    if value is None or isinstance(value, bool):
        raise ValueError("Score value is missing or boolean.")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ValueError("Score value is not numeric.") from exc
    raise ValueError(f"Unsupported score value type: {type(value)}")


def compute_quantile(sorted_scores: list[float], quantile: float) -> float:
    """Return the quantile from a sorted score list (linear interpolation)."""
    quantile_position = (len(sorted_scores) - 1) * quantile
    lower_index = int(math.floor(quantile_position))
    upper_index = int(math.ceil(quantile_position))
    if lower_index == upper_index:
        return sorted_scores[lower_index]
    upper_weight = quantile_position - lower_index
    return sorted_scores[lower_index] * (1 - upper_weight) + sorted_scores[upper_index] * upper_weight


def language_from_path(filepath: str) -> str:
    """Infer the language key from the top-level folder name."""
    parts = Path(filepath).parts
    if not parts:
        raise ValueError(f"Cannot infer language from empty path: {filepath!r}")
    return parts[0]


def list_input_files(
    data_folder: DataFolderLike,
    paths_file: DataFileLike | None,
    recursive: bool,
    glob_pattern: str | None,
) -> list[str]:
    """List input files without sharding to enable language grouping."""
    if paths_file:
        return list(get_shard_from_paths_file(paths_file, 0, 1))
    return data_folder.list_files(recursive=recursive, glob_pattern=glob_pattern)


def group_files_by_language(filepaths: list[str]) -> dict[str, list[str]]:
    """Group filepaths by language directory name."""
    language_directories: dict[str, list[str]] = defaultdict(list)
    for language_file in filepaths:
        language_name = language_from_path(language_file)
        language_directories[language_name].append(language_file)
    return language_directories


def ranked_report_path(report_path: Path, rank: int, world_size: int) -> Path:
    """Write separate report files per rank to avoid clobbering."""
    if world_size <= 1:
        return report_path
    stem = report_path.stem
    suffix = report_path.suffix or ".yaml"
    return report_path.with_name(f"{stem}_rank{rank}{suffix}")


def write_language_report(record: dict, report_path: Path) -> None:
    """Append a YAML document with summary stats for a language."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    score_counts = record["averaged_score_counts"]
    ordered_record = dict(record)
    ordered_record["averaged_score_counts"] = {
        score_value: score_counts[score_value] for score_value in sorted(score_counts)
    }
    with report_path.open("a", encoding="utf-8") as f:
        f.write("---\n")
        yaml.safe_dump(ordered_record, f, sort_keys=False)
