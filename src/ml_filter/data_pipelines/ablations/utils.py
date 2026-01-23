"""Shared helpers for ablation pipelines."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import orjson
from datatrove.data import Document
from datatrove.io import DataFileLike, DataFolderLike, get_shard_from_paths_file
from datatrove.utils.logging import logger
from orjson import JSONDecodeError


def raw_data_adapter(writer, document: Document) -> dict:
    """Return the original JSON payload stored in the document metadata."""
    return document.metadata.get("raw_data", {})


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
    """Return the quantile from a sorted score list (linear interpolation).

    Linear interpolation yields thresholds that remain stable if callers use
    `>` instead of `>=` when filtering.
    """
    quantile_position = (len(sorted_scores) - 1) * quantile
    lower_index = int(math.floor(quantile_position))
    upper_index = int(math.ceil(quantile_position))
    if lower_index == upper_index:
        return sorted_scores[lower_index]
    upper_weight = quantile_position - lower_index
    return sorted_scores[lower_index] * (1 - upper_weight) + sorted_scores[upper_index] * upper_weight


def score_bucket(score: float) -> int:
    """Bucket scores into integer bins 0..5 for reporting."""
    bucket = int(score)
    if bucket < 0:
        return 0
    if bucket > 5:
        return 5
    return bucket


def yaml_string(value: str) -> str:
    """Encode a string as a safe YAML scalar."""
    return orjson.dumps(value).decode("utf-8")


def yaml_value(value) -> str:
    """Encode basic scalar values for YAML output."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return yaml_string(value)
    return str(value)


def language_from_path(filepath: str) -> str:
    """Infer the language key from the top-level folder name."""
    parts = Path(filepath).parts
    if not parts:
        raise ValueError(f"Cannot infer language from empty path: {filepath!r}")
    return parts[0]


def iter_jsonl(data_folder: DataFolderLike, filepath: str, compression: str | None):
    """Yield (line_number, json_obj) for a single JSONL file."""
    with data_folder.open(filepath, "r", compression=compression) as f:
        for line_index, raw_line in enumerate(f):
            try:
                raw = orjson.loads(raw_line)
            except (EOFError, JSONDecodeError) as e:
                logger.warning("Error when reading `%s`: %s", filepath, e)
                continue
            if not isinstance(raw, dict):
                logger.warning("Skipping non-object JSON in `%s` at line %s", filepath, line_index)
                continue
            yield line_index, raw


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


def compute_language_histogram(
    filepaths: list[str],
    score_field: str,
    selection_quantile: float,
    data_folder: DataFolderLike,
    compression: str | None,
):
    """Compute bucket counts and threshold across all files in a language directory."""
    scores: list[float] = []
    total_scored = 0
    score_counts = {i: 0 for i in range(6)}
    for filepath in filepaths:
        for _, row in iter_jsonl(data_folder, filepath, compression):
            if score_field not in row:
                raise ValueError(f"Missing '{score_field}' in {filepath}")
            try:
                score_value = parse_score(row.get(score_field))
            except ValueError as exc:
                raise ValueError(f"Invalid '{score_field}' value in {filepath}: {row.get(score_field)!r}") from exc
            scores.append(score_value)
            score_counts[score_bucket(score_value)] += 1
            total_scored += 1

    if not scores:
        raise ValueError(f"No scores found for score field '{score_field}' in {len(filepaths)} files.")

    scores.sort()
    selection_threshold = compute_quantile(scores, 1.0 - selection_quantile)
    return total_scored, selection_threshold, score_counts


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
    with report_path.open("a", encoding="utf-8") as f:
        f.write("---\n")
        f.write(f"language: {yaml_string(record['language'])}\n")
        f.write(f"file_count: {record['file_count']}\n")
        f.write(f"score_field: {yaml_string(record['score_field'])}\n")
        f.write(f"total_scored: {record['total_scored']}\n")
        f.write(f"selection_quantile: {record['selection_quantile']}\n")
        f.write(f"selection_threshold: {yaml_value(record['selection_threshold'])}\n")
        f.write("score_counts:\n")
        for bucket in range(6):
            f.write(f"  {bucket}: {record['score_counts'][bucket]}\n")
