import json
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping

from loguru import logger

from datatrove.io import DataFileLike, DataFolder, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


class PairedThresholdFilter(BaseDiskReader):
    """Filter *text* JSONL using thresholds computed from paired JSONLs.

    Use-case
    --------
        You have:
            1) A *text* JSONL with fields like: {"document_id": ..., "text": ...}
            2) A *scores* JSONL with fields like: {"document_id": ..., "score_a": ..., ...}
            3) (Optional) A *domains* JSONL with fields like: {"document_id": ..., "domain": ...}

    The folder structures and relative file paths match (same filenames). This reader
    iterates through the scores file and the corresponding text file *in lockstep*.

    Safety checks
    ------------
        For each line index (0-based), we validate:
            - Both files have a line at that index (otherwise error)
            - document_id matches between text and scores line (otherwise error)
            - If domains are enabled: domain line id matches as well (otherwise error)

    If the score thresholds pass, we yield the *text* line as a Document.

    Notes
    -----
    - Filtering is determined from the scores JSONL and (optionally) domains JSONL.
    - The yielded Document content comes from the text JSONL.
    """

    name = "Paired_Filter_By_Threshold"
    _requires_dependencies = []

    def __init__(
        self,
        text_data_folder: DataFolderLike,
        scores_data_folder: DataFolderLike,
        score_keys: Iterable[str],
        thresholds_by_score_key: Mapping[str, float],
        text_jsonl_id_key: str = "document_id",
        score_jsonl_id_key: str = "document_id",
        text_jsonl_text_key: str = "text",
        domains_data_folder: DataFolderLike | None = None,
        accepted_domains: Iterable[str] | None = None,
        domain_jsonl_id_key: str = "document_id",
        domain_jsonl_domain_key: str = "domain",
        compression: Literal["infer", "gzip", "zstd"] | None = None,
        paths_file: DataFileLike | None = None,
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
        # BaseDiskReader will enumerate files from `data_folder`.
        # We want to enumerate using the scores folder, because that's our gating source.
        super().__init__(
            data_folder=scores_data_folder,
            paths_file=paths_file,
            limit=limit,
            skip=skip,
            file_progress=file_progress,
            doc_progress=doc_progress,
            adapter=adapter,
            default_metadata=default_metadata,
            recursive=recursive,
            glob_pattern=glob_pattern,
            shuffle_files=shuffle_files,
        )
        # Normalize folders so we can reliably call .open() regardless of whether
        # the caller passed a str path or a DataFolder instance.
        self._text_data_folder = DataFolder(text_data_folder)
        self._scores_data_folder = DataFolder(scores_data_folder)
        self._score_keys = list(score_keys)
        if len(self._score_keys) == 0:
            raise ValueError("score_keys must contain at least one key")
        self._thresholds_by_score_key = dict(thresholds_by_score_key)
        self._text_jsonl_id_key = text_jsonl_id_key
        self._score_jsonl_id_key = score_jsonl_id_key
        self._text_jsonl_text_key = text_jsonl_text_key
        self._domains_data_folder = DataFolder(domains_data_folder) if domains_data_folder else None
        self._accepted_domains = {str(d) for d in accepted_domains or []}
        self._domain_jsonl_id_key = domain_jsonl_id_key
        self._domain_jsonl_domain_key = domain_jsonl_domain_key
        self._compression = compression
        self._on_mismatch = on_mismatch
        if max_mismatches_per_file < 0:
            raise ValueError("max_mismatches_per_file must be >= 0")
        self._max_mismatches_per_file = max_mismatches_per_file
        if self._domains_data_folder and not self._accepted_domains:
            raise ValueError("accepted_domains must contain at least one value when domains_data_folder is set")

    def read_file(self, filepath: str):
        """Read paired files (scores + text) and yield filtered text Documents."""
        # If a paths_file is used, BaseDiskReader will also read it to enumerate
        # paths. Some Datatrove versions may include that file in the returned
        # shard list as well; skip it defensively.
        if filepath.endswith(".txt"):
            return

        # filepath is relative to scores_data_folder due to BaseDiskReader.
        scores_path = filepath
        text_path = filepath
        domains_path = filepath

        # Ensure the paired text file exists at the same relative path.
        # This gives a clearer error than a generic FileNotFoundError.
        text_abs = Path(str(self._text_data_folder.path)) / text_path
        if not text_abs.exists():
            raise FileNotFoundError(
                "Paired text JSONL file not found for scores JSONL. "
                f"relative_path={text_path!r} expected_text_path={str(text_abs)!r}"
            )

        mismatches = 0

        with self.data_folder.open(scores_path, "r", compression=self._compression) as scores_f:
            with self._text_data_folder.open(text_path, "r", compression=self._compression) as text_f:
                domains_f = None
                if self._domains_data_folder is not None:
                    domains_abs = Path(str(self._domains_data_folder.path)) / domains_path
                    if not domains_abs.exists():
                        raise FileNotFoundError(
                            "Paired domains JSONL file not found for scores JSONL. "
                            f"relative_path={domains_path!r} expected_domains_path={str(domains_abs)!r}"
                        )
                    domains_f = self._domains_data_folder.open(domains_path, "r", compression=self._compression)

                try:
                    iter_rows = zip(scores_f, text_f, domains_f) if domains_f is not None else zip(scores_f, text_f)

                    for li, row in enumerate(iter_rows):
                        if domains_f is None:
                            scores_line, text_line = row
                            domains_line = None
                        else:
                            scores_line, text_line, domains_line = row

                        try:
                            scores_data = json.loads(scores_line)
                            text_data = json.loads(text_line)
                            domains_data = json.loads(domains_line) if domains_line is not None else None
                        except json.JSONDecodeError as e:
                            raise ValueError(
                                f"JSON decode error in paired files at {filepath} line={li}: {e}"
                            )

                        scores_id = scores_data.get(self._score_jsonl_id_key)
                        text_id = text_data.get(self._text_jsonl_id_key)
                        domain_id = None
                        if domains_data is not None:
                            domain_id = domains_data.get(self._domain_jsonl_id_key)

                        if scores_id != text_id or (domains_data is not None and domain_id != text_id):
                            mismatches += 1
                            msg = (
                                f"document_id mismatch in paired files at {filepath} line={li}: "
                                f"scores.{self._score_jsonl_id_key}={scores_id!r} "
                                f"text.{self._text_jsonl_id_key}={text_id!r}"
                            )
                            if domains_data is not None:
                                msg += f" domains.{self._domain_jsonl_id_key}={domain_id!r}"

                            if self._on_mismatch == "raise":
                                raise ValueError(msg)

                            logger.warning(msg)

                            if self._max_mismatches_per_file and mismatches >= self._max_mismatches_per_file:
                                raise ValueError(
                                    f"Too many id mismatches in paired files at {filepath}: "
                                    f"mismatches={mismatches} (max_mismatches_per_file={self._max_mismatches_per_file})"
                                )

                            if self._on_mismatch == "skip_file":
                                return

                            continue

                        score_dict = {k: float(scores_data[k]) for k in self._score_keys if k in scores_data}
                        if not all(k in score_dict for k in self._score_keys):
                            missing = [k for k in self._score_keys if k not in score_dict]
                            raise ValueError(
                                f"Missing score keys {missing} in scores file {scores_path} at line={li}"
                            )

                        domain_value = None
                        if domains_data is not None:
                            domain_value = domains_data.get(self._domain_jsonl_domain_key)
                            if domain_value is None:
                                raise ValueError(
                                    f"Missing domain key {self._domain_jsonl_domain_key!r} in domains file "
                                    f"{domains_path} at line={li}"
                                )
                            if str(domain_value) not in self._accepted_domains:
                                continue

                        if self._passes_thresholds(score_dict):
                            if "file_stem" not in text_data:
                                stem = Path(filepath).name
                                if stem.endswith(".jsonl.gz"):
                                    stem = stem[: -len(".jsonl.gz")]
                                elif stem.endswith(".jsonl"):
                                    stem = stem[: -len(".jsonl")]
                                text_data["file_stem"] = stem

                            text_data.setdefault("file_relpath", filepath)

                            # Add scores and (optional) domain to metadata for output.
                            for key, value in score_dict.items():
                                text_data.setdefault(key, value)
                            if domain_value is not None:
                                text_data.setdefault(self._domain_jsonl_domain_key, domain_value)

                            payload = {
                                "id": text_data.get(self._text_jsonl_id_key),
                                "text": text_data.get(self._text_jsonl_text_key),
                                "metadata": text_data,
                            }
                            doc = self.get_document_from_dict(payload, filepath, li)
                            yield doc

                    extra_scores = next(scores_f, None)
                    extra_text = next(text_f, None)
                    extra_domains = next(domains_f, None) if domains_f is not None else None
                    if extra_scores is not None or extra_text is not None or extra_domains is not None:
                        raise ValueError(
                            f"Line-count mismatch in paired files at {filepath}: "
                            f"scores_has_extra={extra_scores is not None} "
                            f"text_has_extra={extra_text is not None} "
                            f"domains_has_extra={extra_domains is not None}"
                        )
                finally:
                    if domains_f is not None:
                        domains_f.close()

    def _passes_thresholds(self, score_dict: Mapping[str, float]) -> bool:
        for k, threshold in self._thresholds_by_score_key.items():
            if k not in score_dict or score_dict[k] < threshold:
                return False
        return True
