"""Quantile-based filtering for JSONL files grouped by language folders."""

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import orjson
from datatrove.data import Document
from datatrove.io import DataFileLike, DataFolderLike, get_shard_from_paths_file
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.utils.logging import logger
from orjson import JSONDecodeError


def raw_data_adapter(writer, document: Document) -> dict:
    """Return the original JSON payload stored in the document metadata."""
    return document.metadata.get("raw_data", {})


def _parse_score(value) -> float | None:
    """Convert a score value to float if possible."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _compute_quantile(sorted_scores: list[float], q: float) -> float | None:
    """Return the q-quantile from a sorted score list (linear interpolation)."""
    if not sorted_scores:
        return None
    if q <= 0:
        return sorted_scores[0]
    if q >= 1:
        return sorted_scores[-1]
    pos = (len(sorted_scores) - 1) * q
    lower = int(math.floor(pos))
    upper = int(math.ceil(pos))
    if lower == upper:
        return sorted_scores[lower]
    weight = pos - lower
    return sorted_scores[lower] * (1 - weight) + sorted_scores[upper] * weight


def _score_bucket(score: float) -> int:
    """Bucket scores into integer bins 0..5 for reporting."""
    bucket = int(score)
    if bucket < 0:
        return 0
    if bucket > 5:
        return 5
    return bucket


def _yaml_string(value: str) -> str:
    """Encode a string as a safe YAML scalar."""
    return orjson.dumps(value).decode("utf-8")


def _yaml_value(value) -> str:
    """Encode basic scalar values for YAML output."""
    if value is None:
        return "null"
    if isinstance(value, str):
        return _yaml_string(value)
    return str(value)


def _language_from_path(filepath: str) -> str:
    """Infer the language key from the top-level folder name."""
    parts = Path(filepath).parts
    return parts[0] if parts else "unknown"


class QuantileJsonlReader(BaseDiskReader):
    """Read JSONL files and filter rows by per-language quantile threshold."""

    name = "QuantileJsonlReader"
    _requires_dependencies = ["orjson"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        score_field: str,
        selection_quantile: float,
        report_path: Path,
        paths_file: DataFileLike | None = None,
        compression: str | None = "infer",
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter=None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict | None = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder,
            paths_file,
            limit,
            skip,
            file_progress,
            doc_progress,
            adapter,
            text_key,
            id_key,
            default_metadata,
            recursive,
            glob_pattern,
            shuffle_files,
        )
        if selection_quantile < 0 or selection_quantile > 1:
            raise ValueError("selection_quantile must be within [0, 1].")

        self.compression = compression
        self.score_field = score_field
        self.selection_quantile = selection_quantile
        self.report_path = Path(report_path)
        self.source_filename_field = "source_filename"

    def _iter_jsonl(self, filepath: str):
        """Yield (line_number, json_obj) for a single JSONL file."""
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            for li, line in enumerate(f):
                try:
                    raw = orjson.loads(line)
                except (EOFError, JSONDecodeError) as e:
                    logger.warning("Error when reading `%s`: %s", filepath, e)
                    continue
                if not isinstance(raw, dict):
                    logger.warning("Skipping non-object JSON in `%s` at line %s", filepath, li)
                    continue
                yield li, raw

    def _list_files(self) -> list[str]:
        """List all input files without sharding to enable language grouping."""
        if self.paths_file:
            return list(get_shard_from_paths_file(self.paths_file, 0, 1))
        return self.data_folder.list_files(recursive=self.recursive, glob_pattern=self.glob_pattern)

    def _compute_language_histogram(self, filepaths: list[str]):
        """Compute bucket counts and threshold across all files in a language directory."""
        scores: list[float] = []
        total_scored = 0
        score_counts = {i: 0 for i in range(6)}
        for filepath in filepaths:
            for _, row in self._iter_jsonl(filepath):
                if self.score_field not in row:
                    raise ValueError(f"Missing '{self.score_field}' in {filepath}")
                score_value = _parse_score(row.get(self.score_field))
                if score_value is None:
                    raise ValueError(f"Invalid '{self.score_field}' value in {filepath}: {row.get(self.score_field)!r}")
                scores.append(score_value)
                score_counts[_score_bucket(score_value)] += 1
                total_scored += 1

        scores.sort()
        selection_threshold = _compute_quantile(scores, 1.0 - self.selection_quantile)
        return total_scored, selection_threshold, score_counts

    def _ranked_report_path(self, rank: int, world_size: int) -> Path:
        """Write separate report files per rank to avoid clobbering."""
        if world_size <= 1:
            return self.report_path
        stem = self.report_path.stem
        suffix = self.report_path.suffix or ".yaml"
        return self.report_path.with_name(f"{stem}_rank{rank}{suffix}")

    def _write_report(self, record: dict, report_path: Path) -> None:
        """Append a YAML document with summary stats for a language."""
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("a", encoding="utf-8") as f:
            f.write("---\n")
            f.write(f"language: {_yaml_string(record['language'])}\n")
            f.write(f"file_count: {record['file_count']}\n")
            f.write(f"score_field: {_yaml_string(record['score_field'])}\n")
            f.write(f"total_scored: {record['total_scored']}\n")
            f.write(f"selection_quantile: {record['selection_quantile']}\n")
            f.write(f"selection_threshold: {_yaml_value(record['selection_threshold'])}\n")
            f.write("score_counts:\n")
            for bucket in range(6):
                f.write(f"  {bucket}: {record['score_counts'][bucket]}\n")

    def _filter_file(self, filepath: str, selection_threshold: float, language: str) -> Iterable[Document]:
        """Yield documents from a file that meet the selection threshold."""
        for line_idx, row in self._iter_jsonl(filepath):
            if self.score_field not in row:
                raise ValueError(f"Missing '{self.score_field}' in {filepath}")
            score_value = _parse_score(row.get(self.score_field))
            if score_value is None:
                raise ValueError(f"Invalid '{self.score_field}' value in {filepath}: {row.get(self.score_field)!r}")
            if score_value < selection_threshold:
                continue
            doc_payload = dict(row)
            if self.text_key not in doc_payload:
                doc_payload[self.text_key] = "placeholder text"
            document = self.get_document_from_dict(doc_payload, filepath, line_idx)
            if not document:
                continue
            document.metadata["raw_data"] = row
            document.metadata["language"] = language
            if self.source_filename_field not in document.metadata:
                document.metadata[self.source_filename_field] = Path(filepath).stem
            yield document

    def run(self, data: Iterable[Document] = None, rank: int = 0, world_size: int = 1) -> Iterable[Document]:
        """Group files by language, compute thresholds, and stream filtered documents."""
        all_language_files = self._list_files()
        if all_language_files is None or len(all_language_files) == 0:
            raise RuntimeError(f"No files found on {self.data_folder.path}!")

        language_directories: dict[str, list[str]] = defaultdict(list)
        for language_file in all_language_files:
            language_name = _language_from_path(language_file)
            language_directories[language_name].append(language_file)

        all_languages = sorted(language_directories)
        # Deterministic sharding: each rank handles every Nth language.
        languages_for_rank = [lang for i, lang in enumerate(all_languages) if i % world_size == rank]
        report_path = self._ranked_report_path(rank, world_size)

        for language in languages_for_rank:
            filepaths = language_directories[language]
            total_scored, selection_threshold, score_counts = self._compute_language_histogram(filepaths)
            self._write_report(
                {
                    "language": language,
                    "file_count": len(filepaths),
                    "score_field": self.score_field,
                    "total_scored": total_scored,
                    "selection_quantile": self.selection_quantile,
                    "selection_threshold": selection_threshold,
                    "score_counts": score_counts,
                },
                report_path,
            )
            if selection_threshold is None:
                raise ValueError(f"No scores found for language '{language}' in {len(filepaths)} files.")
            for filepath in filepaths:
                for doc in self._filter_file(filepath, selection_threshold, language):
                    self.update_doc_stats(doc)
                    yield doc
