import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Counter, Literal

import numpy as np
from dotenv import load_dotenv

from ml_filter.data_processing.hash_data_files import read_existing_hashes

load_dotenv()

from ml_filter.annotation.regression_head import RegressionHead
from datatrove.pipeline.base import PipelineStep
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.data import DocumentsPipeline, Document
from datatrove.utils.batching import batched
from transformers.utils.hub import cached_file
import dataclasses
import contextlib

from torch import no_grad, cuda, bfloat16

from ml_filter.annotation.embedder import get_embedder_instance
import torch
from typing import Callable
from datatrove.io import DataFileLike, DataFolderLike
from datatrove.pipeline.readers.base import BaseDiskReader
from datatrove.utils.logging import logger
import h5py


def stats_adapter(writer: DiskWriter, document: Document, expand_metadata=True) -> dict:
    """
    The datatrove adapter to write stats metadata without the actual document text
    """
    data = {key: val for key, val in dataclasses.asdict(document).items() if val and key != "text"}
    if writer.expand_metadata and "metadata" in data:
        data |= data.pop("metadata")
    return data

#TODO modify this function based on csv lookup table
def _get_unique_id(doc: Document, filepath: str) -> str:
    return f"{filepath}_{doc.id}"

def _get_file_path(doc: Document) -> str:
    base_name = os.path.basename(doc.metadata.get("file_path", "default.jsonl"))
    filepath = os.path.splitext(base_name)[0]
    return filepath


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
        self.hash_map = read_existing_hashes(csv_hashmap)

    def read_file(self, filepath: str):
        import orjson
        from orjson import JSONDecodeError

        with self.data_folder.open(filepath, "r", compression=self.compression) as f:
            try:
                full_file_path = str(self.data_folder.path) + "/" + filepath
                #log out self.data_folder
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

        for doc_batch in batched(doc_pipeline, self.batch_size):
            with self.track_time(unit='batch'):
                embeddings = embedder.embed([doc.text for doc in doc_batch])
                for idx, (doc, embedding) in enumerate(zip(doc_batch, embeddings)):
                    base_name = os.path.basename(doc.metadata.get("file_path", "default.jsonl"))
                    filepath = os.path.splitext(base_name)[0]
                    doc.metadata["source_filename"] = _get_file_path(doc)
                    # Convert tensor to list for JSON serialization
                    doc.metadata['embedding'] = embedding.cpu().tolist()
                    #print each embedding and document_id
                    logger.info(f"Document ID: {doc.metadata['document_id']}")
                    yield doc


class JQLHead(PipelineStep):
    """
    A pipeline step that applies one or more regression heads to document embeddings to produce scalar scores.

    This step supports batch processing and distributed device assignment. Each regression head outputs a
    separate score, which is added to the document metadata under a prefixed key (e.g., "score_Edu-JQL-Mistral-SF").

    If no custom regression head checkpoints are provided, the default JQL Edu regression heads for Gemma, Mistral,
    and Llama (trained on Snowflake embeddings) are used.

    Args:
        regression_head_checkpoints: a dictionary mapping head names to checkpoint paths. If None, uses default heads.
        batch_size: number of documents to process in one batch. Default is 1000.
        device_overwrite: manually specify a CUDA device (e.g., "0"). If None, devices are assigned per rank automatically.
        stats_writer: optional writer to record scored documents for analysis or debugging.
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
            logger.info('No custom regression heads specified. Using default JQL Edu heads.')
            self.regression_head_checkpoints = {
                'Edu-JQL-Gemma-SF': cached_file('Jackal-AI/JQL-Edu-Heads',
                                                'checkpoints/edu-gemma-snowflake-balanced.ckpt'),
                'Edu-JQL-Mistral-SF': cached_file('Jackal-AI/JQL-Edu-Heads',
                                                  'checkpoints/edu-mistral-snowflake-balanced.ckpt'),
                'Edu-JQL-Llama-SF': cached_file('Jackal-AI/JQL-Edu-Heads',
                                                'checkpoints/edu-llama-snowflake-balanced.ckpt'),
            }
        else:
            self.regression_head_checkpoints = regression_head_checkpoints
        self.batch_size = batch_size
        self.device_overwrite = device_overwrite
        self.stats_writer = stats_writer

    def run(self, doc_pipeline: DocumentsPipeline, rank: int = 0, world_size: int = 1, **kwargs) -> DocumentsPipeline:
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

        with self.stats_writer if self.stats_writer else contextlib.nullcontext() as writer:
            for doc_batch in batched(doc_pipeline, self.batch_size):
                # breakpoint()
                with self.track_time(unit='batch'):
                    # Convert embeddings back to tensors with bfloat16 dtype
                    embeddings = [torch.tensor(doc.text, device=device, dtype=torch.bfloat16) for doc in doc_batch]
                    # embeddings = [torch.tensor(doc.metadata['embedding'], device=device, dtype=torch.bfloat16) for doc in doc_batch]
                    embeddings_tensor = torch.stack(embeddings)

                    scores = {}
                    with no_grad():
                        for name, regression_head in self.regression_heads.items():
                            scores[f'score_{name}'] = regression_head(embeddings_tensor).cpu().squeeze(1)

                    for batch_idx, doc in enumerate(doc_batch):
                        base_name = os.path.basename(doc.metadata.get("file_path", "default.jsonl"))
                        filepath = os.path.splitext(base_name)[0]
                        doc.metadata["source_filename"] = filepath
                        for name, score in scores.items():
                            doc.metadata[name] = score[batch_idx].item()
                        if writer:
                            writer.write(doc, rank)
                        yield doc


class JQLEmbeddingReader(BaseDiskReader):
    """
    Read data from HDF5 (.h5) files containing precomputed embeddings and associated labels.

    Each HDF5 file is expected to contain a dataset group (default: 'train') with:
      - "embeddings": an array of embedding vectors
      - "labels": an array of corresponding label values (can be scalars or vectors)

    Each embedding-label pair is treated as a document, where:
      - "embedding" is stored as the main data
      - "labels" are added to the metadata
      - "id" is generated as the sample index

    Args:
        data_folder: a str, tuple, or DataFolder object representing a path/filesystem.
        dataset_name: name of the dataset group inside each HDF5 file (default: "embeddings", overridden to "train" internally).
        paths_file: optionally provide a file listing relative file paths to load, one per line.
        limit: limit the number of documents read in total. Useful for debugging or sampling.
        skip: number of initial documents to skip.
        file_progress: show a progress bar across files being read.
        doc_progress: show a progress bar for documents within each file.
        text_key: the key used for the main data field in the document. Default is "embedding".
        adapter: optional function to adapt the raw dict to a document. Signature: (self, data: dict, path: str, id_in_file: int | str).
        id_key: the key used to identify each document. Default is "id".
        default_metadata: dictionary of metadata to include in every document.
        recursive: whether to scan subdirectories of `data_folder` for files.
        glob_pattern: only files matching this pattern will be included (default: "*.h5").
        shuffle_files: if True, shuffles the list of files before reading. Mostly for visualization/debugging.
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
            text_key: str = "embedding",
            adapter: Callable = None,
            id_key: str = "id",
            default_metadata: dict = None,
            recursive: bool = False,
            glob_pattern: str | None = "*.h5",
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
        Open .h5 file in SWMR-safe read-only mode. Yield docs with embedding.
        """
        try:

            with self.data_folder.open(filepath, "rb") as fs_file:
                with h5py.File(fs_file) as f:

                    dataset_name = "train"

                    if dataset_name not in f:
                        raise KeyError(f"Dataset '{dataset_name}' not found in {filepath}")

                    grp = f[dataset_name]
                    embeddings = torch.from_numpy(grp["embeddings"][:]).float()
                    labels = torch.from_numpy(grp["labels"][:]).float()

                    if len(embeddings) != len(labels):
                        raise ValueError(
                            f"Mismatched number of embeddings and labels in {filepath}: "
                            f"{len(embeddings)} embeddings vs {len(labels)} labels"
                        )

                    n_samples = len(embeddings)

                    logger.info(f"Dataset '{dataset_name}' has {n_samples} samples in {filepath}")

                    for i in range(n_samples):
                        with self.track_time():
                            doc_dict = {
                                "id": str(i),
                                "embedding": embeddings[i].tolist(),
                                "labels": labels[i].tolist(),
                            }
                            yield self.get_document_from_dict(doc_dict, filepath, i)

        except Exception as e:
            logger.warning(f"Failed to read `{filepath}`: {e}")



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
        max_file_size: int = 5 * 2**30,  # 5GB
        schema: Any = None,  # Optional, not used in h5 but kept for compatibility
        dataset_name: str = "train",  # Default dataset name
    ):
        super().__init__(
            output_folder,
            output_filename,
            compression=None,  # No compression here
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

        #TODO double check
        if group_name in file:
            del file[group_name]

        group = file.create_group(group_name)
        group.create_dataset("embeddings", data=embeddings, compression="gzip", dtype=np.float32)
        dt = h5py.string_dtype(encoding='utf-8')
        group.create_dataset("document_id", data=document_id, compression="gzip", dtype=dt)


    def _write(self, document: dict, file_handler, filename: str):
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