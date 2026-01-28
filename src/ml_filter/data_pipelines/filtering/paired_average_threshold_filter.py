from __future__ import annotations

from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping

from datatrove.io import DataFolderLike

from ml_filter.data_pipelines.filtering.paired_threshold_filter import PairedThresholdFilter


class PairedAverageThresholdFilter(PairedThresholdFilter):
    """Filter *text* JSONL using averaged thresholds across score keys.

    This reader is identical to PairedThresholdFilter, except it computes the
    average of all configured score keys and compares it against a single
    threshold (optionally overridden per top-level folder).
    """

    name = "Paired_Filter_By_Average_Threshold"

    def __init__(
        self,
        text_data_folder: DataFolderLike,
        scores_data_folder: DataFolderLike,
        score_keys: Iterable[str],
        average_threshold: float | None,
        average_thresholds_by_folder: Mapping[str, float] | None = None,
        text_jsonl_id_key: str = "document_id",
        score_jsonl_id_key: str = "document_id",
        text_jsonl_text_key: str = "text",
        domains_data_folder: DataFolderLike | None = None,
        accepted_domains: Iterable[str] | None = None,
        domain_jsonl_id_key: str = "document_id",
        domain_jsonl_domain_key: str = "domain",
        compression: Literal["infer", "gzip", "zstd"] | None = None,
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable | None = None,
        default_metadata: dict | None = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
        on_mismatch: Literal["raise", "skip_line", "skip_file"] = "raise",
        max_mismatches_per_file: int = 0,
    ):
        super().__init__(
            text_data_folder=text_data_folder,
            scores_data_folder=scores_data_folder,
            score_keys=score_keys,
            thresholds_by_score_key={},
            thresholds_by_folder=None,
            text_jsonl_id_key=text_jsonl_id_key,
            score_jsonl_id_key=score_jsonl_id_key,
            text_jsonl_text_key=text_jsonl_text_key,
            domains_data_folder=domains_data_folder,
            accepted_domains=accepted_domains,
            domain_jsonl_id_key=domain_jsonl_id_key,
            domain_jsonl_domain_key=domain_jsonl_domain_key,
            compression=compression,
            limit=limit,
            skip=skip,
            file_progress=file_progress,
            doc_progress=doc_progress,
            adapter=adapter,
            default_metadata=default_metadata,
            recursive=recursive,
            glob_pattern=glob_pattern,
            shuffle_files=shuffle_files,
            on_mismatch=on_mismatch,
            max_mismatches_per_file=max_mismatches_per_file,
        )
        self._average_threshold = average_threshold
        self._average_thresholds_by_folder = {
            str(folder): float(threshold)
            for folder, threshold in (average_thresholds_by_folder or {}).items()
        }
        if self._average_threshold is None and not self._average_thresholds_by_folder:
            raise ValueError(
                "average_threshold must be provided when average_thresholds_by_folder is empty."
            )

    def _passes_thresholds(self, score_dict: Mapping[str, float], filepath: str) -> bool:
        threshold = self._average_threshold
        if self._average_thresholds_by_folder:
            folder = Path(filepath).parts[0] if filepath else None
            if folder in self._average_thresholds_by_folder:
                threshold = self._average_thresholds_by_folder[folder]

        if threshold is None:
            raise ValueError(
                "Missing average threshold for folder. Provide average_threshold or "
                "include the folder in average_thresholds_by_folder."
            )

        average_score = sum(score_dict[k] for k in self._score_keys) / len(self._score_keys)
        return average_score >= threshold
