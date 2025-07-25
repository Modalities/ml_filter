# --- Standard Library ---
import contextlib
import dataclasses
import os
from pathlib import Path
from typing import Callable, Optional

import h5py
import numpy as np
import torch
from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFileLike, DataFolderLike
from collections import Counter, defaultdict

from typing import Any, Callable, Literal, Optional, IO, Union

from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.batching import batched
from datatrove.utils.logging import logger
from dotenv import load_dotenv

### Debugging
import torch

from ml_filter.annotation.embedder import (
    GteMultilingualBase,
    JinaEmbeddingsV3TextMatching,
    SnowflakeArcticEmbedMV2_0,
    get_embedder_instance,
)
from ml_filter.data_processing.hash_data_files import read_existing_hashes

import os

import torch
import gc
import dataclasses
from torch import bfloat16, cuda, no_grad
import time
from ml_filter.annotation.regression_head import RegressionHead

load_dotenv()



def find_max_batch_size_annotation(model, embedding_dim=768, initial_batch=16, max_batch=1_000_000_000_000, device='cuda', dtype=torch.bfloat16, warmup_iters=2, measure_iters=5):
    """
    Finds the largest batch size that fits in memory and measures throughput at each step.

    Args:
        model (torch.nn.Module): Model to test (should be on the correct device).
        embedding_dim (int): Input embedding dimensionality.
        initial_batch (int): Starting batch size.
        max_batch (int): Upper limit for batch size.
        device (str): Device to test on ('cuda' or 'cpu').
        dtype (torch.dtype): Data type of input.
        warmup_iters (int): Number of warm-up iterations before measuring.
        measure_iters (int): Number of iterations to measure throughput.

    Returns:
        Tuple[int, float]: Largest batch size and its corresponding throughput (samples/sec).
    """
    low = initial_batch
    high = max_batch
    best = 0
    best_throughput = 0.0

    while low <= high:
        batch_size = (low + high) // 2
        try:
            print(f"\nTesting batch size: {batch_size}")
            dummy_input = torch.randn(batch_size, embedding_dim, device=device, dtype=dtype)

            # Warm-up iterations (ignore timing)
            with torch.no_grad():
                for _ in range(warmup_iters):
                    _ = model(dummy_input)

            # Timed iterations
            torch.cuda.synchronize()
            start = time.time()
            with torch.no_grad():
                for _ in range(measure_iters):
                    _ = model(dummy_input)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            throughput = (batch_size * measure_iters) / elapsed
            print(f"✅ Batch size {batch_size} succeeded. Throughput: {throughput:.2f} samples/sec")

            # Update best values
            best = batch_size
            best_throughput = throughput
            low = batch_size + 1

        except torch.cuda.OutOfMemoryError:
            print(f"❌ OOM at batch size {batch_size}. Trying smaller batch...")
            high = batch_size - 1

        except Exception as e:
            print(f"⚠️  Unexpected error at batch size {batch_size}: {e}")
            break

        finally:
            torch.cuda.empty_cache()
            gc.collect()

    print(f"\nOptimal batch size: {best / 1_000_000:.2f}M, Throughput: {best_throughput / 1_000_000:.2f}M samples/sec")
    return best, best_throughput


def stats_adapter(writer: DiskWriter, document: Document, expand_metadata=True) -> dict:
    """
    Adapter function for extracting metadata from a Document object, excluding the 'text' field.

    This is typically used in evaluation or statistics pipelines where only metadata such as
    IDs, scores, and document properties are needed for storage or further analysis.

    Byte values in metadata are decoded to UTF-8 to ensure safe serialization.

    Args:
        writer (DiskWriter): The disk writer handling output.
        document (Document): The document to extract metadata from.
        expand_metadata (bool): Whether to flatten the nested 'metadata' dict into the top level.

    Returns:
        dict: A metadata dictionary safe for JSON serialization, without the document text.
    """

    def safe_json(val):
        if isinstance(val, bytes):
            return val.decode('utf-8', errors='ignore')  # or base64 if binary
        return val

    data = {key: safe_json(val) for key, val in dataclasses.asdict(document).items() if val and key != "text"}
    if writer.expand_metadata and "metadata" in data:
        metadata = data.pop("metadata")
        metadata = {k: safe_json(v) for k, v in metadata.items()}
        data |= metadata
    return data


def find_max_batch_size(
    embedder: Union[GteMultilingualBase, SnowflakeArcticEmbedMV2_0, JinaEmbeddingsV3TextMatching],
    doc_pipeline: DocumentsPipeline,
    max_limit: int = 5000,
    step: int = 100,
) -> int:
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
            batch = docs[:batch_size]
            texts = [doc.text for doc in batch]
            _ = embedder.embed(texts)
            logger.info(f"✅ Batch size {batch_size} succeeded")
            max_batch_size = batch_size
            batch_size += step
            torch.cuda.empty_cache()
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"❌ OOM at batch size {batch_size}, reducing step")
                if step <= 1:
                    break
                batch_size -= step
                step = max(1, step // 2)
                batch_size += step
            else:
                raise e
        finally:
            torch.cuda.empty_cache()

    logger.info(f"🏁 Max batch size found: {max_batch_size}")
    return max_batch_size


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
                            if not document:
                                continue
                            document.metadata["file_path"] = full_file_path
                            document.metadata["document_id"] = file_hash + "_" + str(li)
                            document.metadata["source_filename"] = Path(full_file_path).relative_to(self.data_folder.path)
                        except (EOFError, JSONDecodeError) as e:
                            logger.warning(f"Error when reading `{filepath}`: {e}")
                            continue
                    yield document
            except UnicodeDecodeError as e:
                logger.warning(f"File `{filepath}` may be corrupted: raised UnicodeDecodeError ({e})")


def _get_file_path(doc: Document) -> str:
    base_name = os.path.basename(doc.metadata.get("file_path", "default.jsonl"))
    filepath = os.path.splitext(base_name)[0]
    return filepath

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
        embedder_model_id: str = "Snowflake/snowflake-arctic-embed-m-v2.0",
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
        if not cuda.is_available():
            logger.warning("CUDA is not available, using CPU")
            device = "cpu"
        else:
            if self.device_overwrite is None:
                device_count = cuda.device_count()
                cuda_device_id = rank % device_count
                device = f"cuda:{cuda_device_id}"
            else:
                device = f"cuda:{self.device_overwrite}"

        embedder = get_embedder_instance(self.embedder_model_id, device, bfloat16)

        # Find the maximum batch size that fits in GPU memory
        # Uncomment the next line to find the max batch size dynamically
        # self.batch_size = find_max_batch_size(doc_pipeline, embedder, max_limit=1000000, step=500)

        total_docs = 0
        total_time = 0

        with self.stats_writer if self.stats_writer else contextlib.nullcontext() as writer:
            for doc_batch in batched(doc_pipeline, self.batch_size):
                with self.track_time(unit="batch"):
                    start_time = time.time()
                    try:
                        embeddings = embedder.embed([doc.text for doc in doc_batch])
                        for idx, (doc, embedding) in enumerate(zip(doc_batch, embeddings)):
                            doc.metadata["embedding"] = embedding
                            if writer:
                                writer.write(doc, rank)
                            yield doc

                        duration = time.time() - start_time
                        num_docs = len(doc_batch)
                        throughput = num_docs / duration if duration > 0 else 0

                        total_time += duration
                        total_docs += num_docs
                        average_throughput = total_docs / total_time if total_time > 0 else 0

                        logger.info(
                            f"Processed {num_docs} docs in {duration:.2f}s in rank {rank} → Throughput: {throughput:.2f} docs/sec"
                        )
                        logger.info(f"Average throughput: {average_throughput:.2f} docs/sec")
                        print(
                            f"Processed {num_docs} docs in {duration:.2f}s in rank {rank} → Throughput: {throughput:.2f} docs/sec"
                        )
                        print(f"Average throughput: {average_throughput:.2f} docs/sec")
                        torch.cuda.empty_cache()

                    except RuntimeError as e:
                        if "out of memory" in str(e):
                            logger.error(
                                f"CUDA OOM error on rank {rank} with batch size {self.batch_size}. "
                                f"Consider reducing the batch size or using a smaller model."
                            )
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
        max_file_size: int = 5 * 2**30,
        schema: Any = None,
        dataset_name: str = "train",
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
        embeddings = np.stack([doc["metadata"]["embedding"] for doc in batch], dtype=np.float32)
        document_id = [doc["metadata"]["document_id"] for doc in batch]

        file = self._writers[filename]
        group_name = self.dataset_name

        if group_name not in file:
            group = file.create_group(group_name)
            maxshape_emb = (None, embeddings.shape[1])
            maxshape_ids = (None,)
            group.create_dataset(
                "embeddings", data=embeddings, maxshape=maxshape_emb, compression="gzip", dtype=np.float32
            )
            dt = h5py.string_dtype(encoding="utf-8")
            group.create_dataset("document_id", data=document_id, maxshape=maxshape_ids, compression="gzip", dtype=dt)
        else:
            group = file[group_name]
            emb_ds = group["embeddings"]
            id_ds = group["document_id"]

            # Current sizes
            old_size = emb_ds.shape[0]
            new_size = old_size + embeddings.shape[0]

            # Resize datasets to hold new batch
            emb_ds.resize(new_size, axis=0)
            id_ds.resize(new_size, axis=0)

            # Append new data
            emb_ds[old_size:new_size, :] = embeddings
            id_ds[old_size:new_size] = document_id

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
        logger.info("#### SUCCESS #####")
        super().close()



class JQLEmbeddingReader(BaseDiskReader):
    """
    A specialized DiskReader that reads HDF5 (.h5) files containing precomputed embeddings.

    Each .h5 file is expected to contain a group (e.g., 'train') with two datasets:
    - 'embeddings': a 2D array of floats representing document embeddings
    - 'document_id': a list of document identifiers

    This reader converts each embedding into a Document object with optional tracking and metadata.

    Args:
        data_folder (DataFolderLike): Folder or archive containing the HDF5 files.
        dataset_name (str): Name of the dataset group within each .h5 file.
        paths_file (DataFileLike, optional): A file listing which files to read.
        limit (int): Maximum number of documents to read (-1 for no limit).
        skip (int): Number of documents to skip initially.
        file_progress (bool): Whether to show progress for files.
        doc_progress (bool): Whether to show progress for documents.
        text_key (str): The field name used for embeddings.
        adapter (Callable, optional): Optional adapter to transform raw doc dicts.
        id_key (str): The field name to use as document ID.
        default_metadata (dict, optional): Default metadata to attach to each document.
        recursive (bool): Whether to recursively search for files.
        glob_pattern (str, optional): File-matching pattern (default: '*.h5').
        shuffle_files (bool): Whether to shuffle the order of files read.
    """

    name = "🔢 JQL-EMBEDDING-READER"
    _requires_dependencies = ["h5py", "numpy"]

    def __init__(
        self,
        data_folder: DataFolderLike,
        dataset_name: str = "embeddings",
        paths_file: DataFileLike | None = None,
        limit: int = -1,
        skip: int = 0,
        file_progress: bool = True,
        doc_progress: bool = True,
        text_key: str = "embeddings",
        adapter: Callable = None,
        id_key: str = "document_id",
        default_metadata: dict = None,
        recursive: bool = True,
        glob_pattern: str | None = "**/*.h5",
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
            id_key=id_key,
            text_key=text_key,
            default_metadata=default_metadata,
            recursive=recursive,
            glob_pattern=glob_pattern,
            shuffle_files=shuffle_files,
        )
        self.dataset_name = dataset_name

    def read_file(self, filepath: str):
        """
        Reads a single .h5 file and yields document objects with embeddings.

        Expects the specified dataset group to contain 'embeddings' and 'document_id'.
        Converts them into a document stream for downstream pipeline steps.

        Args:
            filepath (str): Path to the .h5 file.

        Yields:
            Document: A document object containing an embedding and its metadata.

        Raises:
            KeyError: If the expected dataset group is not found.
            ValueError: If the number of embeddings and IDs do not match.
        """
        try:

            with self.data_folder.open(filepath, "rb") as fs_file:
                with h5py.File(fs_file) as f:

                    dataset_name = "train"

                    if dataset_name not in f:
                        raise KeyError(f"Dataset '{dataset_name}' not found in {filepath}")

                    grp = f[dataset_name]
                    embeddings = torch.from_numpy(grp["embeddings"][:]).float()
                    document_ids = grp["document_id"][:]

                    if len(embeddings) != len(document_ids):
                        raise ValueError(
                            f"Mismatched number of embeddings and labels in {filepath}: "
                            f"{len(embeddings)} embeddings vs {len(document_ids)} labels"
                        )

                    n_samples = len(embeddings)

                    logger.info(f"Dataset '{dataset_name}' has {n_samples} samples in {filepath}")

                    for i in range(n_samples):
                        with self.track_time():
                            doc_dict = {
                                "id": str(i),
                                "embeddings": embeddings[i].tolist(),
                                "document_id": document_ids[i],
                            }
                            doc = self.get_document_from_dict(doc_dict, filepath, i)
                            doc.metadata["document_id"] = document_ids[i].decode('utf-8')
                            doc.metadata["source_filename"] = str(Path(doc.metadata.get("file_path")).relative_to(self.data_folder.path))
                            yield doc

        except Exception as e:
            logger.warning(f"Failed to read `{filepath}`: {e}")


class JQLHead(PipelineStep):
    """
    A pipeline step that applies one or more regression heads to document embeddings to generate scores.

    This step supports evaluating large batches of precomputed embeddings by applying
    fine-tuned regression models (e.g., for educational scoring, quality estimation, etc.).
    The output scores are stored as new metadata fields in each document.

    Args:
        regression_head_checkpoints (dict[str, str], optional): Mapping of head names to checkpoint paths.
            If None, defaults to JQL educational scoring heads from the Jackal-AI hub.
        batch_size (int): Number of documents to process in a single batch.
        device_overwrite (str, optional): Manually specify CUDA device (e.g., '0' or 'cuda:1').
        stats_writer (DiskWriter, optional): Optional writer to log or save document scores.
    """
    name = "🔢 - JQL-HEAD"
    type = "🔢 - JQL-HEAD"

    def __init__(
        self,
        regression_head_checkpoints: Optional[dict[str, str]] = None,
        batch_size: int = 1_000,
        device_overwrite: Optional[str] = None,
        stats_writer: DiskWriter = None,
    ):
        super().__init__()
        if regression_head_checkpoints is None:
            raise ValueError("No regression head checkpoints provided. Please specify custom regression heads.")
        self.regression_head_checkpoints = regression_head_checkpoints

        self.batch_size = batch_size
        self.device_overwrite = device_overwrite
        self.stats_writer = stats_writer

    def run(self, doc_pipeline: DocumentsPipeline, rank: int = 0, world_size: int = 1, **kwargs) -> DocumentsPipeline:
        """
        Applies regression heads to document embeddings and yields documents with updated scores.

        Handles device selection (CPU or CUDA), loads the regression heads, and writes scores
        into document metadata fields like 'score_Edu-JQL-Gemma-SF'.

        Args:
            doc_pipeline (DocumentsPipeline): Iterable pipeline of documents with embedding data in `text`.
            rank (int): Rank ID in distributed processing (used for device assignment).
            world_size (int): Total number of distributed workers (not currently used).

        Yields:
            Document: Each document enriched with predicted scores in its metadata.
        """
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

        self.regression_heads = {}
        for name, path in self.regression_head_checkpoints.items():
            self.regression_heads[name] = RegressionHead.load_from_checkpoint(path, map_location=device).to(bfloat16)

        # self.batch_size = find_max_batch_size_annotation(next(iter(self.regression_heads.values())))[0]

        total_docs = 0
        total_time = 0
        # log batch size.
        logger.info(f"Using batch size: {self.batch_size} for rank {rank}")

        with self.stats_writer if self.stats_writer else contextlib.nullcontext() as writer:
            for doc_batch in batched(doc_pipeline, self.batch_size):
                with self.track_time(unit="batch"):
                    start_time = time.time()
                    # Convert embeddings back to tensors with bfloat16 dtype
                    embeddings = [torch.tensor(doc.text, device=device, dtype=torch.bfloat16) for doc in doc_batch]
                    embeddings_tensor = torch.stack(embeddings)

                    scores = {}
                    with no_grad():
                        for name, regression_head in self.regression_heads.items():
                            scores[f'score_{name}'] = regression_head(embeddings_tensor).cpu().squeeze(1)

                    for batch_idx, doc in enumerate(doc_batch):
                        for name, score in scores.items():
                            doc.metadata[name] = score[batch_idx].item()
                        if writer:
                            writer.write(doc, rank)
                        yield doc
                    
                    duration = time.time() - start_time
                    num_docs = len(doc_batch)
                    throughput = num_docs / duration if duration > 0 else 0

                    total_time += duration
                    total_docs += num_docs
                    average_throughput = total_docs / total_time if total_time > 0 else 0

                    logger.info(
                        f"Processed {num_docs} docs in {duration:.2f}s in rank {rank} → Throughput: {throughput:.2f} docs/sec"
                    )
                    logger.info(f"Average throughput: {average_throughput:.2f} docs/sec")
                    print(
                        f"Processed {num_docs} docs in {duration:.2f}s in rank {rank} → Throughput: {throughput:.2f} docs/sec"
                    )
                    print(f"Average throughput: {average_throughput:.2f} docs/sec")
                    torch.cuda.empty_cache()
