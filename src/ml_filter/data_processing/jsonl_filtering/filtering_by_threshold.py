import json
from pathlib import Path
from typing import Callable, Iterable, Literal, Mapping

from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader


class ScoresParser(BaseDiskReader):
    """
    A parser that reads a JSONL file containing scores for samples. 
    Each entry in the JSONL file is expected to have a "document_id" field, 
    and the scores for that sample. A threshold value is provided for each score key,
    and samples with scores above the threshold are what I need.
    """

    name = "Filter_By_Threshold"
    _requires_dependencies = []



    def __init__(
        self,
        data_folder: DataFolderLike,
        score_keys: Iterable[str],
        thresholds_by_score_key: Mapping[str, float],
        compression: Literal["infer", "gzip", "zstd"] | None = None,
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
        self._thresholds_by_score_key = dict(thresholds_by_score_key)
        self._compression = compression

    def read_file(self, filepath: str):
        """
        Reads a JSONL file and yields Document objects for each line that passes the threshold(s).
        Only yields lines where all required score_keys are present and pass the thresholds.
        Args:
            filepath: path of the file to read
        Yields:
            Document: for each line passing the filter
        """
        stem = Path(filepath).name
        if stem.endswith(".jsonl.gz"):
            stem = stem[: -len(".jsonl.gz")]
        elif stem.endswith(".jsonl"):
            stem = stem[: -len(".jsonl")]

        relpath = filepath

        with self.data_folder.open(filepath, "r", compression=self._compression) as f:
            for li, line in enumerate(f):
                try:
                    file_data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                # Only keep if all score_keys are present
                if not all(k in file_data for k in self._score_keys):
                    continue
                score_dict = {k: float(file_data[k]) for k in self._score_keys}
                if self._passes_thresholds(score_dict):
                    # Make filename templating easier for writers.
                    if isinstance(file_data, dict):
                        file_data.setdefault("file_stem", stem)
                        file_data.setdefault("file_relpath", relpath)
                    document = self.get_document_from_dict(file_data, filepath, li)
                    yield document





    def _passes_thresholds(self, score_dict: Mapping[str, float]) -> bool:
        """
        Return True iff all configured threshold keys meet their threshold.
        Only keys present in thresholds_by_score_key are used for filtering.
        """
        for k, threshold in self._thresholds_by_score_key.items():
            if k not in score_dict or score_dict[k] < threshold:
                return False
        return True



