# --- Standard Library ---
import gc
import os
from collections import Counter, defaultdict
import time
from pathlib import Path
from typing import Any, Callable, Literal, Optional, IO

# --- Third-Party Libraries ---
import h5py
import numpy as np
import torch
# --- Project-Specific Imports ---
from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.batching import batched
from datatrove.utils.logging import logger
from dotenv import load_dotenv
from torch import bfloat16, cuda

from ml_filter.annotation.embedder import get_embedder_instance
from ml_filter.data_processing.hash_data_files import read_existing_hashes

import dataclasses
import contextlib

load_dotenv()




def find_max_batch_size(embedder, doc_pipeline, max_limit=5000, step=100):
    """
    Finds the max batch size that doesn't cause CUDA OOM.

    Args:
        embedder: your embedder instance with `.embed()` method
        doc_pipeline: iterable or list of documents with `.text`
        max_limit: upper bound for batch size search
        step: increment step for searching

    Returns:
        int: max batch size that fits in GPU memory without OOM
    """
    batch_size = step
    max_batch_size = 0

    docs = list(doc_pipeline)

    while batch_size <= max_limit:
        try:
            batch = docs[:batch_size]
            texts = [doc.text for doc in batch]

            # # Clear memory before inference
            del batch
            gc.collect()
            torch.cuda.empty_cache()

            batch = docs[:batch_size]
            texts = [doc.text for doc in batch]

            _ = embedder.embed(texts, batch_size)

            logger.info(f"✅ Batch size {batch_size} succeeded")
            max_batch_size = batch_size
            batch_size += step

        except RuntimeError as e:
            if 'out of memory' in str(e):
                logger.warning(f"❌ OOM at batch size {batch_size}, reducing step")
                torch.cuda.empty_cache()
                gc.collect()
                if step <= 1:
                    break
                batch_size -= step
                step = max(1, step // 2)
                batch_size += step
            else:
                raise e

        finally:
            # Free references every iteration
            if 'batch' in locals(): del batch
            if 'texts' in locals(): del texts
            gc.collect()
            torch.cuda.empty_cache()
            ...

    logger.info(f"🏁 Max batch size found: {max_batch_size}")
    return max_batch_size


def stats_adapter(writer: DiskWriter, document: Document, expand_metadata=True) -> dict:
    """
    The datatrove adapter to write stats metadata without the actual document text

    Args:
        writer: the diskwriter
        document: a datatrove document

    Returns: a dictionary of metadata without the text field

    """
    data = {key: val for key, val in dataclasses.asdict(document).items() if val and key != "text"}
    if writer.expand_metadata and "metadata" in data:
            data |= data.pop("metadata")
    return data

def _get_file_path(doc: Document) -> str:
    """
    Extract a clean filename (without extension) from a document's metadata for consistent source tracking.

    This is commonly used to associate documents with their original file source in a normalized format.

    Args:
        doc (Document): The document whose 'file_path' metadata will be parsed.

    Returns:
        str: The base filename without extension. Defaults to 'default' if no file_path is present.
    """
    return Path(doc.metadata.get("file_path", "default.jsonl")).stem


class JQLJsonlReader(BaseDiskReader):
    """Read data from JSONL files.
        Will read each line as a separate document.

    Args:
        data_folder: a str, tuple or DataFolder object representing a path/filesystem
        paths_file: optionally provide a file with one path per line (without the `data_folder` prefix) to read.
        compression: the compression to use (default: "infer")
        limit: limit the number of documents to read. Useful for debugging
        skip: skip the first n rows
        file_progress: show progress bar for files
        doc_progress: show progress bar for documents
        adapter: function to adapt the data dict from the source to a Document.
            Takes as input: (self, data: dict, path: str, id_in_file: int | str)
                self allows access to self.text_key and self.id_key
            Returns: a dict with at least a "text" and "id" keys
        text_key: the key containing the text data (default: "text").
        id_key: the key containing the id for each sample (default: "id").
        default_metadata: a dictionary with any data that should be added to all samples' metadata
        recursive: whether to search files recursively. Ignored if paths_file is provided
        glob_pattern: pattern that all files must match exactly to be included (relative to data_folder). Ignored if paths_file is provided
        shuffle_files: shuffle the files within the returned shard. Mostly used for data viz. purposes, do not use with dedup blocks
    """

    name = "🐿 Jsonl"
    _requires_dependencies = ["orjson"]

    def __init__(
            self,
            data_folder: DataFolderLike,
            csv_hashmap: Path,
            paths_file: DataFileLike | None = None,
            compression: Literal["infer", "gzip", "zstd"] | None = "infer",
            limit: int = -1,
            skip: int = 0,
            file_progress: bool = False,
            doc_progress: bool = False,
            adapter: Callable = None,
            text_key: str = "text",
            id_key: str = "id",
            default_metadata: dict = None,
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
        try:
            self.hash_map = read_existing_hashes(csv_hashmap)
        except FileNotFoundError:
            logger.error(f"Hash CSV file not found at path: {csv_hashmap}")
            raise
        except Exception as e:
            logger.error(f"Failed to load hash map from {csv_hashmap}: {e}")
            raise

    def read_file(self, filepath: str):
        import orjson
        from orjson import JSONDecodeError

        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            try:
                full_file_path = str(self.data_folder.path) + "/" + filepath
                logger.info("Reading file %s", full_file_path)
                logger.info("data folder %s", self.data_folder.path)
                file_hash = self.hash_map[full_file_path]
                for li, line in enumerate(f):
                    with self.track_time():
                        try:
                            document = self.get_document_from_dict(orjson.loads(line), filepath, li)
                            document.metadata['file_path'] = full_file_path
                            document.metadata['document_id'] = file_hash + "_" + str(li)
                            if not document:
                                continue
                        except (EOFError, JSONDecodeError) as e:
                            logger.warning(f"Error when reading `{filepath}`: {e}")
                            continue
                    yield document
            except UnicodeDecodeError as e:
                logger.warning(f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({e})")


class JQLEmbedder(PipelineStep):
    """
    A pipeline step that embeds batches of documents using a specified embedding model.

    This step supports GPU acceleration and handles device assignment across multiple ranks
    (e.g., in distributed settings). Each document receives an `embedding` field in metadata
    containing the embedding vector.

    Args:
        embedder_model_id: HuggingFace model ID or local path to the embedding model. Default is Snowflake Arctic embed v2.0.
        batch_size: number of documents to process in one embedding batch. Default is 1000.
        device_overwrite: manually specify a CUDA device (e.g., "0"). If None, a device is selected automatically per rank.
        stats_writer: optional writer to log stats to disk during embedding.
    """
    name = "🔢 - JQL-EMBEDDER"
    type = "🔢 - JQL-EMBEDDER"

    def __init__(
            self,
            embedder_model_id: str = 'Snowflake/snowflake-arctic-embed-m-v2.0',
            batch_size: int = 1_000,
            device_overwrite: Optional[str] = None,
            stats_writer: DiskWriter = None,

    ):
        super().__init__()
        self.embedder_model_id = embedder_model_id
        self.batch_size = batch_size
        self.device_overwrite = device_overwrite
        self.stats_writer = stats_writer

    def run(self, doc_pipeline: DocumentsPipeline, rank: int = 0, world_size: int = 1, **kwargs) -> DocumentsPipeline:
        torch.cuda.memory._record_memory_history(max_entries=100000)
        if not cuda.is_available():
            logger.warning('CUDA is not available, using CPU')
            device = 'cpu'
        else:
            if self.device_overwrite is None:
                device_count = cuda.device_count()
                cuda_device_id = rank % device_count
                device = f'cuda:{cuda_device_id}'
            else:
                device = f'cuda:{self.device_overwrite}'

        embedder = get_embedder_instance(self.embedder_model_id, device, bfloat16)

        # self.batch_size = find_max_batch_size(embedder, doc_pipeline, max_limit=1000000, step=500)

        for doc_batch in batched(doc_pipeline, self.batch_size):
            with self.track_time(unit='batch'):
                start_time = time.time()

                try:
                    embeddings = embedder.embed([doc.text for doc in doc_batch])

                    for idx, (doc, embedding) in enumerate(zip(doc_batch, embeddings)):
                        doc.metadata["source_filename"] = _get_file_path(doc)
                        doc.metadata['embedding'] = embedding
                        yield doc

                    duration = time.time() - start_time
                    num_docs = len(doc_batch)
                    throughput = num_docs / duration if duration > 0 else 0
                    logger.info(f"Processed {num_docs} docs in {duration:.2f}s → Throughput: {throughput:.2f} docs/sec")
                    print(torch.cuda.max_memory_allocated())
                    torch.cuda.empty_cache()
                    gc.collect()
                    torch.cuda.memory._dump_snapshot("snapshot_tok.pickle")

                    for obj in gc.get_objects():
                        try:
                            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                                if obj.is_cuda:
                                    print(
                                        f"Tensor: {type(obj)}, Size: {obj.size()}, Memory: {obj.element_size() * obj.nelement() / 1024 ** 2:.2f} MB")
                        except Exception as e:
                            pass

                except RuntimeError as e:
                    if 'out of memory' in str(e):
                        logger.error(f"CUDA OOM error on rank {rank} with batch size {self.batch_size}. "
                                     f"Consider reducing the batch size or using a smaller model.")
                        print(torch.cuda.memory_summary())
                        print(torch.cuda.max_memory_allocated())
                        raise e
                    else:
                        logger.error(f"Runtime error on rank {rank}: {e}")
                        raise e



class HDF5Writer(DiskWriter):
    """
    A writer that stores batched documents with embeddings into HDF5 (.h5) files.

    This writer accumulates documents in memory and periodically flushes them to disk
    once `batch_size` is reached, storing them in a group called `dataset_name` within
    the HDF5 file.

    Each group contains:
      - "embeddings": a 2D NumPy array of float32 embedding vectors
      - "document_id": a list of UTF-8 encoded string identifiers

    Args:
        output_folder: where the HDF5 files will be written (can be a path or DataFolder).
        output_filename: optional base filename to write to. May be ignored if file rotation occurs.
        adapter: optional callable to adapt documents before writing.
        batch_size: number of documents to accumulate before writing to file.
        expand_metadata: whether to expand nested metadata dicts (not used here).
        max_file_size: maximum file size in bytes before rotating to a new file (default: 5GB).
        schema: not used in HDF5 writer, included for compatibility with other writers.
        dataset_name: name of the group inside each HDF5 file where datasets will be written (default: "train").
    """
    default_output_filename: str = None
    name = "💾 HDF5"
    _requires_dependencies = ["h5py"]

    def __init__(
            self,
            output_folder: DataFolderLike,
            output_filename: str = None,
            adapter: Callable = None,
            batch_size: int = 1000,
            expand_metadata: bool = False,
            max_file_size: int = 5 * 2 ** 30,
            schema: Any = None,
            dataset_name: str = "train"
    ):
        super().__init__(
            output_folder,
            output_filename,
            compression=None,
            adapter=adapter,
            mode="wb",
            expand_metadata=expand_metadata,
            max_file_size=max_file_size,
        )
        self._writers = {}
        self._batches = defaultdict(list)
        self._file_counter = Counter()
        self.batch_size = batch_size
        self.schema = schema
        self.dataset_name = dataset_name

    def _write_batch(self, filename: str):
        if not self._batches[filename]:
            return

        batch = self._batches.pop(filename)

        logger.info(f"the structure of document is {batch[0]}")
        embeddings = np.stack([doc["metadata"]["embedding"] for doc in batch], dtype=np.float32)
        document_id = [doc["metadata"]["document_id"] for doc in batch]

        file = self._writers[filename]

        group_name = self.dataset_name

        if group_name in file:
            del file[group_name]

        group = file.create_group(group_name)
        group.create_dataset("embeddings", data=embeddings, compression="gzip", dtype=np.float32)
        dt = h5py.string_dtype(encoding='utf-8')
        group.create_dataset("document_id", data=document_id, compression="gzip", dtype=dt)

    def _write(self, document: dict, file_handler: IO, filename: str):
        if filename not in self._writers:
            self._writers[filename] = h5py.File(file_handler.name, "a")

        self._batches[filename].append(document)

        if len(self._batches[filename]) >= self.batch_size:
            self._write_batch(filename)

    def close(self):
        for filename in list(self._batches):
            self._write_batch(filename)
        for f in self._writers.values():
            f.close()
        self._writers.clear()
        self._batches.clear()
        super().close()
