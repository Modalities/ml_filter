import json
import logging
from pathlib import Path
from typing import Callable, Iterable, Literal

from datatrove.data import DocumentsPipeline
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


class ScoresParser(BaseDiskReader):
    """
    A parser that reads a JSONL file containing scores for samples and maps them to the
    corresponding tokenized data files. Each entry in the JSONL file is expected to have
    a "document_id" field that contains a base file hash and an index, and the scores
    for that sample.
    """

    name = "ScoresParser"
    # type = "Parser"
    _requires_dependencies = []

    SCORE_ENTRIES_KEY = "score_entries"
    TOKENIZED_FILE_KEY = "tokenized_file"

    def __init__(
        self,
        data_folder: DataFolderLike,
        score_keys: Iterable[str],
        tokenized_data_path: Path,
        base_file_prefix: Path = Path(""),
        tokenized_data_extension: str = ".pbin",
        compression: Literal["infer", "gzip", "zstd"] | None = "infer",
        paths_file: DataFileLike | None = None,
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = False,
        doc_progress: bool = False,
        adapter: Callable | None = None,
        text_key: str = "text",
        id_key: str = "id",
        default_metadata: dict | None = None,
        recursive: bool = True,
        glob_pattern: str | None = None,
        shuffle_files: bool = False,
    ):
        super().__init__(
            data_folder=data_folder,
            paths_file=paths_file,
            limit=limit,
            skip=skip,
            file_progress=file_progress,
            doc_progress=doc_progress,
            adapter=adapter,
            text_key=text_key,
            id_key=id_key,
            default_metadata=default_metadata,
            recursive=recursive,
            glob_pattern=glob_pattern,
            shuffle_files=shuffle_files,
        )
        self._score_keys = list(score_keys)
        assert len(self._score_keys) > 0, "At least one score key must be provided."
        self._tokenized_data_path = tokenized_data_path
        self._base_file_prefix = base_file_prefix
        self._tokenized_data_extension = tokenized_data_extension
        self._compression = compression

    def read_file(self, filepath: str) -> DocumentsPipeline:
        """
        Turns a given JSONL file into a Document object containing the path to the corresponding tokenized data file
        and a list of dictionaries with the scores for each sample in the file.
        Args:
            filepath: path of the file to read

        Returns: generator of Document
        """
        base_file_path_or_name, scores_as_list = self._parse_scores_jsonl_file(filepath)
        tokenized_data_path = self._map_to_tokenized_data_path(base_file_path_or_name)
        doc_content = {
            "text": ".",  # Text needs to be non-empty.
            self.SCORE_ENTRIES_KEY: scores_as_list,
            self.TOKENIZED_FILE_KEY: tokenized_data_path,
        }
        document = self.get_document_from_dict(doc_content, filepath, 0)
        return [document]

    def _parse_scores_jsonl_file(self, filepath: str) -> tuple[str, list[dict[str, float]]]:
        scores_for_document_idx: dict[str, dict[str, float]] = {}
        processed_count = 0
        duplicate_counts: dict[str, int] = {}  # track counts per original document_id

        with self.data_folder.open(filepath, "r", compression=self._compression) as f:
            for line_number, line in enumerate(f, start=1):
                processed_count += 1
                file_data = json.loads(line)
                document_id = file_data.get("document_id")

                if document_id in scores_for_document_idx:
                    # Generate a new unique ID with a numeric suffix to disambiguate duplicates.
                    dup_count = duplicate_counts.get(document_id, 0) + 1
                    duplicate_counts[document_id] = dup_count
                    # Use underscore + count; ensure no collision with an existing (unlikely but safe guard if previous had suffix already)
                    new_id = f"{document_id}_{dup_count}"
                    while new_id in scores_for_document_idx:
                        dup_count += 1
                        duplicate_counts[document_id] = dup_count
                        new_id = f"{document_id}_{dup_count}"
                    print(
                        f"Duplicate document_id '{document_id}' encountered at line {line_number} in {filepath}. Renamed to '{new_id}'."
                    )
                    document_id = new_id

                scores_for_document_idx[document_id] = {k: float(file_data[k]) for k in self._score_keys}

            self._verify_unique_ids(filepath, scores_for_document_idx, processed_count)
            scores_as_list = [scores for _, scores in sorted(scores_for_document_idx.items(), key=lambda x: x[0])]
            return f.name, scores_as_list

    def _verify_unique_ids(self, filepath: str, scores_for_document_idx: dict[str, dict], processed_count: int):
        """Verify that the number of unique document IDs matches the number of processed (valid) lines.

        Args:
            filepath: Path to the scores JSONL file.
            scores_for_document_idx: Mapping of document_id to its score dict.
            processed_count: Number of lines that contained a valid document_id.
        """
        unique_ids = len(scores_for_document_idx)
        if unique_ids != processed_count:
            raise ValueError(
                f"Mismatch in number of samples in scores file {filepath}: unique_ids={unique_ids} processed_lines={processed_count}."
            )


    def _map_to_tokenized_data_path(self, base_file_path: Path | str) -> Path:
        """
        Maps a base file path to the corresponding tokenized data path.
        Args:
            base_file_path (str): The path of the base file.
        Returns:
            Path: The path to the tokenized data file.
        """
        if isinstance(base_file_path, str):
            base_file_path = Path(base_file_path)

        # When prefix is effectively empty ("" or ".") just take the file name.
        if str(self._base_file_prefix) in {"", "."}:
            base_name = base_file_path.name  # ensure we only use the filename portion
            base_file_rel = Path(base_name)
        else:
            # Use relative_to only if possible; otherwise fall back to filename.
            try:
                base_file_rel = base_file_path.relative_to(self._base_file_prefix)
            except Exception:
                base_file_rel = Path(base_file_path.name)

        tokenized_rel = base_file_rel.with_suffix(self._tokenized_data_extension)
        tokenized_data_path = self._tokenized_data_path / tokenized_rel
        if not tokenized_data_path.exists():
            raise FileNotFoundError(f"Tokenized data file {tokenized_data_path} does not exist.")
        return tokenized_data_path
