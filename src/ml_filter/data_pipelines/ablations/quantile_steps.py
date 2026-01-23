"""Quantile-based reporting for JSONL files grouped by language folders."""

from pathlib import Path
from typing import Iterable

from datatrove.data import Document
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader

from .utils import (
    compute_language_histogram,
    group_files_by_language,
    list_input_files,
    ranked_report_path,
    write_language_report,
)


class QuantileJsonlReader(BaseDiskReader):
    """Read JSONL files and report per-language quantile thresholds."""

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

    def read_file(self, filepath: str) -> Iterable[Document]:
        """BaseDiskReader API; use run() which computes per-language thresholds."""
        raise RuntimeError("QuantileJsonlReader requires per-language thresholds; use run() instead of read_file().")

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
        languages_for_rank = [lang for i, lang in enumerate(all_languages) if i % world_size == rank]
        report_path = ranked_report_path(self.report_path, rank, world_size)

        for language in languages_for_rank:
            filepaths = language_directories[language]
            total_scored, selection_threshold, score_counts = compute_language_histogram(
                filepaths=filepaths,
                score_field=self.score_field,
                selection_quantile=self.selection_quantile,
                data_folder=self.data_folder,
                compression=self.compression,
            )
            write_language_report(
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
        return iter(())
