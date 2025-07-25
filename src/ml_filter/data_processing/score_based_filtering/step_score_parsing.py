import json
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
        hash_to_base_file: dict[str, Path],
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
        self._hash_to_base_file = hash_to_base_file
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
        base_file_hash, scores_as_list = self._parse_scores_jsonl_file(filepath)
        tokenized_data_path = self._map_to_tokenized_data_path(base_file_hash)
        doc_content = {
            "text": ".",  # Text needs to be non-empty.
            self.SCORE_ENTRIES_KEY: scores_as_list,
            self.TOKENIZED_FILE_KEY: tokenized_data_path,
        }
        document = self.get_document_from_dict(doc_content, filepath, 0)
        return [document]

    def _parse_scores_jsonl_file(self, filepath: str) -> tuple[str, list[dict[str, float]]]:
        scores_for_idx: dict[int, dict[str, float]] = {}
        hashes: set[str] = set()
        with self.data_folder.open(filepath, "r", compression=self._compression) as f:
            for line in f:
                file_data = json.loads(line)
                base_file_hash, document_idx = file_data["document_id"].rsplit("_", 1)
                scores_for_idx[int(document_idx)] = {k: file_data[k] for k in self._score_keys}
                hashes.add(base_file_hash)
        self._verify_file_format(scores_for_idx, hashes)
        scores_as_list = list(map(lambda x: x[1], sorted(scores_for_idx.items(), key=lambda x: x[0])))
        base_file_hash = next(iter(hashes))
        return base_file_hash, scores_as_list

    def _verify_file_format(self, scores_for_idx: dict[int, dict[str, float]], hashes: set[str]):
        assert len(hashes) == 1, "All entries in the score file must refer to the same base file."
        assert min(scores_for_idx.keys()) == 0 and max(scores_for_idx.keys()) + 1 == len(
            scores_for_idx
        ), "All indices in the score file must be continuous."

    def _map_to_tokenized_data_path(self, base_file_hash: str) -> Path:
        """
        Maps a base file hash to the corresponding tokenized data path.
        Args:
            base_file_hash (str): The hash of the base file.
        Returns:
            Path: The path to the tokenized data file.
        """
        if base_file_hash not in self._hash_to_base_file:
            raise ValueError(f"Base file hash {base_file_hash} not found in the provided hash mapping.")
        base_file = self._hash_to_base_file[base_file_hash]
        base_file_rel = base_file.relative_to(self._base_file_prefix)
        tokenized_rel = base_file_rel.with_suffix(self._tokenized_data_extension)
        tokenized_data_path = self._tokenized_data_path / tokenized_rel
        if not tokenized_data_path.exists():
            raise FileNotFoundError(f"Tokenized data file {tokenized_data_path} does not exist.")
        return tokenized_data_path
