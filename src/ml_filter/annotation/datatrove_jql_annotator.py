import contextlib
import dataclasses
import os
from pathlib import Path
from typing import Callable, Optional

import h5py
import torch
from datatrove.data import Document, DocumentsPipeline
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.utils.batching import batched
from datatrove.utils.logging import logger
from dotenv import load_dotenv
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


def _get_file_path(doc: Document) -> str:
    base_name = os.path.basename(doc.metadata.get("file_path", "default.jsonl"))
    filepath = os.path.splitext(base_name)[0]
    return filepath


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

        self.batch_size = find_max_batch_size_annotation(next(iter(self.regression_heads.values())))[0]

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
