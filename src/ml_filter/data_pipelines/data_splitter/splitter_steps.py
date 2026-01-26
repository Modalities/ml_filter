"""Split JSONL files into score buckets per language."""

from pathlib import Path
from typing import Iterable

import orjson
from datatrove.data import Document
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader
from orjson import JSONDecodeError

from ml_filter.data_pipelines.ablations.utils import group_files_by_language, list_input_files

from .utils import average_score, bucket_index, normalize_buckets, validate_bucket_template


class BucketedJsonlSplitter(BaseDiskReader):
    """Read JSONL files and write rows into bucketed JSONL files by language."""

    name = "BucketedJsonlSplitter"
    _requires_dependencies = ["orjson"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        score_fields: list[str],
        buckets: list[tuple[float, float]],
        output_dir: Path,
        output_subdir: str = "data_buckets",
        bucket_filename_template: str = "{lower}_{upper}_bucked_{language}.jsonl",
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
        self.compression = compression
        self.score_fields = score_fields
        self.buckets = normalize_buckets(buckets)
        self.output_dir = Path(output_dir)
        self.output_subdir = output_subdir
        self.bucket_filename_template = bucket_filename_template
        validate_bucket_template(bucket_filename_template)

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
        """Required by BaseDiskReader but not used by BucketedJsonlSplitter."""
        raise NotImplementedError("BucketedJsonlSplitter does not support read_file().")

    def run(self, data: Iterable[Document] = None, rank: int = 0, world_size: int = 1) -> Iterable[Document]:
        """Group files by language and split rows into bucketed JSONL files."""
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
        languages_for_rank = [lang for i, lang in enumerate(all_languages) if i % world_size == rank]

        for language in languages_for_rank:
            filepaths = language_directories[language]
            writers = {}
            try:
                for filepath in filepaths:
                    for _, row in self._iter_jsonl(filepath):
                        average = average_score(row, self.score_fields, filepath)
                        bucket = bucket_index(average, self.buckets)
                        if bucket is None:
                            continue
                        _, _, lower_label, upper_label = self.buckets[bucket]
                        bucket_filename = self.bucket_filename_template.format(
                            lower=lower_label,
                            upper=upper_label,
                            language=language,
                        )
                        bucket_path = self.output_dir / language / self.output_subdir / bucket_filename
                        writer = writers.get(bucket_path)
                        if writer is None:
                            bucket_path.parent.mkdir(parents=True, exist_ok=True)
                            writer = bucket_path.open("a", encoding="utf-8")
                            writers[bucket_path] = writer
                        writer.write(orjson.dumps(row).decode("utf-8"))
                        writer.write("\n")
            finally:
                for writer in writers.values():
                    writer.close()
        return iter(())
