"""Quantile-based reporting for JSONL files grouped by language folders."""

from pathlib import Path
from typing import Iterable

import orjson
from datatrove.data import Document
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader
from orjson import JSONDecodeError

from .utils import (
    compute_quantile,
    group_files_by_language,
    list_input_files,
    parse_score,
    ranked_report_path,
    write_language_report,
)


class QuantileJsonlReader(BaseDiskReader):
    """Read JSONL files and report per-language quantile thresholds for averaged scores."""

    name = "QuantileJsonlReader"
    _requires_dependencies = ["orjson"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        score_fields: list[str],
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
        self.score_fields = score_fields
        self.selection_quantile = selection_quantile
        self.report_path = Path(report_path)
        self.source_filename_field = "source_filename"

    def _iter_jsonl(self, filepath: str):
        """Read a JSONL file and yield each row as a dict with its line index."""
        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            for line_index, raw_line in enumerate(f):
                try:
                    raw = orjson.loads(raw_line)
                except (EOFError, JSONDecodeError) as exc:
                    raise ValueError(f"Invalid JSON in `{filepath}` at line {line_index}") from exc
                if not isinstance(raw, dict):
                    raise ValueError(f"Expected JSON object in `{filepath}` at line {line_index}")
                yield line_index, raw

    def read_file(self, filepath: str) -> Iterable[Document]:
        """Required by BaseDiskReader but not used by QuantileJsonlReader."""
        raise NotImplementedError("QuantileJsonlReader does not support read_file().")

    def _compute_language_histogram(self, filepaths: list[str]):
        """Compute bucket counts and threshold using averaged per-row scores."""
        average_scores: list[float] = []
        total_rows_scored = 0
        score_value_counts: dict[float, int] = {}
        for filepath in filepaths:
            for _, row in self._iter_jsonl(filepath):
                missing_fields = [field for field in self.score_fields if field not in row]
                if missing_fields:
                    raise ValueError(f"Missing score fields {missing_fields} in {filepath}")
                try:
                    row_scores = [parse_score(row.get(field)) for field in self.score_fields]
                except ValueError as exc:
                    raise ValueError(f"Invalid score value in {filepath} for fields {self.score_fields}") from exc
                average_score = sum(row_scores) / len(row_scores)
                average_scores.append(average_score)
                score_value_counts[average_score] = score_value_counts.get(average_score, 0) + 1
                total_rows_scored += 1

        if total_rows_scored == 0:
            raise ValueError(f"No rows with score fields {self.score_fields} found in {len(filepaths)} files.")

        average_scores.sort()
        selection_threshold = compute_quantile(average_scores, 1.0 - self.selection_quantile)
        return total_rows_scored, selection_threshold, score_value_counts

    def run(self, data: Iterable[Document] = None, rank: int = 0, world_size: int = 1) -> Iterable[Document]:
        """Group files by language, compute thresholds, and write reports."""
        all_language_files = list_input_files(
            data_folder=self.data_folder,
            paths_file=self.paths_file,
            recursive=self.recursive,
            glob_pattern=self.glob_pattern,
        )
        if all_language_files is None or len(all_language_files) == 0:
            raise RuntimeError(f"No files found on {self.data_folder.path}!")

        language_directories = group_files_by_language(all_language_files)

        all_languages = sorted(language_directories)
        # Deterministic sharding: each rank handles every Nth language.
        # Each rank handles every Nth language, starting at its rank index.
        languages_for_rank = [lang for i, lang in enumerate(all_languages) if i % world_size == rank]
        report_path = ranked_report_path(self.report_path, rank, world_size)

        for language in languages_for_rank:
            filepaths = language_directories[language]
            total_rows_scored, quantile_threshold, score_value_counts = self._compute_language_histogram(filepaths)
            write_language_report(
                {
                    "language": language,
                    "file_count": len(filepaths),
                    "score_fields": self.score_fields,
                    "total_rows_scored": total_rows_scored,
                    "selection_quantile": self.selection_quantile,
                    "selection_threshold": quantile_threshold,
                    "averaged_score_counts": score_value_counts,
                },
                report_path,
            )
        return iter(())
