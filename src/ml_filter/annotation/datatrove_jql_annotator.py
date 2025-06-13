import os
from collections import defaultdict
from pathlib import Path
from typing import Optional, List, Dict, Any

import numpy as np
from dotenv import load_dotenv

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


def _get_unique_id(doc: Document, filepath: str) -> str:
    return f"{filepath}_{doc.id}"

def _get_file_path(doc: Document) -> str:
    base_name = os.path.basename(doc.metadata.get("file_path", "default.jsonl"))
    filepath = os.path.splitext(base_name)[0]
    return filepath

class JQLEmbedder(PipelineStep):
    """
    A pipeline step for embedding text documents using a specified embedding model.
    """
    name = "🔢 - JQL-EMBEDDER"
    type = "🔢 - JQL-EMBEDDER"

    def __init__(
            self,
            embedder_model_id: str = 'Snowflake/snowflake-arctic-embed-m-v2.0',
            batch_size: int = 1_000,
            device_overwrite: Optional[str] = None,
    ):
        super().__init__()
        self.embedder_model_id = embedder_model_id
        self.batch_size = batch_size
        self.device_overwrite = device_overwrite

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
                    # Convert tensor to list for JSON serialization
                    doc.metadata['embedding'] = embedding.cpu().tolist()
                    doc.metadata['document_id'] = _get_unique_id(doc, _get_file_path(doc))
                    yield doc


class JQLHead(PipelineStep):
    """
    A pipeline step for applying regression heads to embeddings to produce scores.
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
    Read data from an HDF5 (.h5) file containing precomputed embeddings.
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


class HDF5EmbeddingWriter(PipelineStep):
    """
    HDF5 writer that creates one HDF5 file per input JSONL file.
    Each source JSONL file gets its own corresponding HDF5 file.
    """

    def __init__(self,
                 output_folder: str | Path,
                 dataset_name: str = "embeddings",
                 batch_size: int = 10000,
                 embedding_key: str = "embedding",
                 compression: str = "gzip",
                 chunk_size: int = 1000,
                 overwrite_existing: bool = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.output_folder = Path(output_folder)
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.embedding_key = embedding_key
        self.compression = compression
        self.chunk_size = chunk_size
        self.overwrite_existing = overwrite_existing

        # Per-source file tracking
        self.source_buffers = defaultdict(lambda: {
            'embeddings': [],
            'document_ids': [],
            'metadata': [],
            'total_written': 0,
            'initialized': False,
            'embedding_dim': None
        })

        # Ensure output directory exists
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def _get_source_key(self, document: Document) -> str:
        """Extract source file identifier from document metadata."""
        # Try different possible source keys that datatrove might use
        for key in ['source', 'source_file', 'filename', 'file_path']:
            if key in document.metadata:
                source_path = Path(document.metadata[key])
                return source_path.stem  # filename without extension

        # Fallback: try to extract from document ID if it contains path info
        if hasattr(document, 'id') and '/' in document.id:
            return Path(document.id).stem

        # Last resort: use a default key
        logger.warning(
            f"Could not determine source file for document {getattr(document, 'id', 'unknown')}, using 'unknown'")
        return "unknown"

    def _get_output_path(self, source_key: str, rank: int = 0) -> Path:
        """Get the HDF5 output path for a given source file."""
        if rank > 0:
            filename = f"{source_key}_rank_{rank}.h5"
        else:
            filename = f"{source_key}.h5"
        return self.output_folder / filename

    def _initialize_source_file(self, source_key: str, rank: int):
        """Initialize HDF5 file for a specific source."""
        file_path = self._get_output_path(source_key, rank)
        buffer = self.source_buffers[source_key]

        # If overwriting, delete file
        if file_path.exists() and self.overwrite_existing:
            try:
                file_path.unlink()
                logger.info(f"Deleted existing file {file_path} due to overwrite_existing.")
                raise ValueError(
                    f"File {file_path} already exists and overwrite_existing is set to True. "
                    "Please check your configuration."
                )
            except Exception as e:
                logger.error(f"Failed to delete {file_path}: {e}")
                raise

        # If not overwriting, check for existing dataset to resume
        if file_path.exists() and not self.overwrite_existing:
            try:
                with h5py.File(file_path, "r") as f:
                    if self.dataset_name in f:
                        grp = f[self.dataset_name]
                        existing = grp["embeddings"].shape[0]
                        dim = grp["embeddings"].shape[1]
                        buffer['embedding_dim'] = dim
                        buffer['total_written'] = existing
                        logger.info(f"Resuming source '{source_key}' dataset in {file_path} "
                                    f"with {existing} samples, dim={dim}")
            except OSError:
                logger.warning(f"Cannot open existing file {file_path}; it will be overwritten.")
                try:
                    file_path.unlink()
                except:
                    pass

        buffer['initialized'] = True

    def run(self, data: DocumentsPipeline, rank: int = 0, world_size: int = 1) -> DocumentsPipeline:
        """Process documents from the pipeline and accumulate embeddings by source file."""

        # Iterate through the DocumentsPipeline (generator of Document objects)
        for document in data:
            with self.track_time():
                self._process_document(document, rank)
            yield document

        # When done processing all documents, flush remaining buffers
        self._flush_all_buffers(rank)

    def _process_document(self, document: Document, rank: int):
        """Process a single document."""
        try:
            # Check if document has metadata attribute
            if not hasattr(document, 'metadata') or document.metadata is None:
                logger.warning(f"Document {getattr(document, 'id', 'unknown')} has no metadata")
                raise ValueError("Document metadata is missing or None")

            # Extract embedding
            if self.embedding_key not in document.metadata:
                logger.warning(
                    f"No embedding key '{self.embedding_key}' for document {getattr(document, 'id', 'unknown')}")
                logger.debug(f"Available metadata keys: {list(document.metadata.keys())}")
                raise ValueError(f"Embedding key '{self.embedding_key}' not found in document metadata")

        except Exception as e:
            logger.error(f"Error accessing document metadata: {e}")
            logger.error(f"Document type: {type(document)}")
            raise ValueError(f"Invalid document metadata for {getattr(document, 'id', 'unknown')}: {e}")

        # Determine which source file this document belongs to
        source_key = self._get_source_key(document)
        buffer = self.source_buffers[source_key]

        # Initialize this source file if needed
        if not buffer['initialized']:
            self._initialize_source_file(source_key, rank)

        # Extract embedding
        emb = document.metadata[self.embedding_key]
        if isinstance(emb, list):
            emb = np.array(emb, dtype=np.float32)
        elif not isinstance(emb, np.ndarray):
            logger.warning(f"Invalid embedding type for {document.id}: {type(emb)}")
            return
        else:
            emb = emb.astype(np.float32)

        # Flatten if needed
        if emb.ndim != 1:
            emb = emb.reshape(-1)

        # Check/set embedding dimension for this source
        if buffer['embedding_dim'] is None:
            buffer['embedding_dim'] = emb.shape[0]
            logger.info(f"Set embedding_dim={emb.shape[0]} for source '{source_key}'")
        elif emb.shape[0] != buffer['embedding_dim']:
            raise ValueError(f"Embedding dim mismatch for doc {document.id} in source '{source_key}': "
                             f"expected {buffer['embedding_dim']}, got {emb.shape[0]}")

        # Add to this source's buffers
        buffer['embeddings'].append(emb)
        buffer['document_ids'].append(document.id)
        meta = {k: v for k, v in document.metadata.items() if k != self.embedding_key}
        buffer['metadata'].append(meta)

        # Check if we should flush this source's buffer
        if len(buffer['embeddings']) >= self.batch_size:
            self._flush_source_buffer(source_key, rank)

    def _flush_source_buffer(self, source_key: str, rank: int):
        """Flush buffer for a specific source file."""
        buffer = self.source_buffers[source_key]

        if not buffer['embeddings']:
            return

        embeddings = np.vstack(buffer['embeddings'])
        doc_ids = list(buffer['document_ids'])
        metadata = list(buffer['metadata'])

        self._append_to_hdf5(source_key, embeddings, doc_ids, metadata, rank)

        buffer['total_written'] += embeddings.shape[0]
        buffer['embeddings'].clear()
        buffer['document_ids'].clear()
        buffer['metadata'].clear()

        logger.info(f"Flushed {embeddings.shape[0]} embeddings for source '{source_key}' "
                    f"(total this run: {buffer['total_written']})")

    def _flush_all_buffers(self, rank: int):
        """Flush all remaining buffers when pipeline ends."""
        for source_key in self.source_buffers:
            self._flush_source_buffer(source_key, rank)
            buffer = self.source_buffers[source_key]
            if buffer['initialized']:
                logger.info(f"Finished source '{source_key}'; total written: {buffer['total_written']}")

    def _append_to_hdf5(self,
                        source_key: str,
                        embeddings: np.ndarray,
                        document_ids: List[str],
                        metadata: List[Dict[str, Any]],
                        rank: int):
        """Append embeddings to HDF5 file for specific source."""
        file_path = self._get_output_path(source_key, rank)
        buffer = self.source_buffers[source_key]
        new_n = embeddings.shape[0]

        with h5py.File(file_path, "a") as f:
            if self.dataset_name in f:
                # Append branch
                grp = f[self.dataset_name]
                # Check embedding dim
                existing_dim = grp["embeddings"].shape[1]
                if existing_dim != buffer['embedding_dim']:
                    raise ValueError(f"Embedding dimension mismatch in '{source_key}': "
                                     f"existing {existing_dim}, new {buffer['embedding_dim']}")
                old_n = grp["embeddings"].shape[0]
                total_n = old_n + new_n
                grp["embeddings"].resize((total_n, buffer['embedding_dim']))
                grp["embeddings"][old_n:total_n] = embeddings

                dt = h5py.string_dtype(encoding='utf-8')
                grp["document_ids"].resize((total_n,))
                grp["document_ids"][old_n:total_n] = np.array(document_ids, dtype=dt)

                grp["metadata"].resize((total_n,))
                # Serialize metadata or placeholders
                meta_strs = []
                for m in metadata:
                    try:
                        meta_strs.append(json.dumps(m))
                    except Exception:
                        logger.warning("Failed to serialize metadata; storing {}")
                        meta_strs.append("{}")
                if len(meta_strs) < new_n:
                    meta_strs.extend(["{}"] * (new_n - len(meta_strs)))
                grp["metadata"][old_n:total_n] = np.array(meta_strs, dtype=dt)

                grp.attrs["n_samples"] = total_n
                logger.debug(f"Appended {new_n} samples to '{source_key}' dataset in {file_path} (now {total_n})")

            else:
                # Create branch
                grp = f.create_group(self.dataset_name)
                chunk_rows = self.chunk_size

                # embeddings
                grp.create_dataset(
                    "embeddings",
                    data=embeddings,
                    maxshape=(None, buffer['embedding_dim']),
                    chunks=(chunk_rows, buffer['embedding_dim']),
                    compression=self.compression,
                    dtype=np.float32
                )

                # document_ids
                dt = h5py.string_dtype(encoding='utf-8')
                grp.create_dataset(
                    "document_ids",
                    data=np.array(document_ids, dtype=dt),
                    maxshape=(None,),
                    chunks=(chunk_rows,),
                    dtype=dt,
                    compression=self.compression
                )

                # metadata
                meta_strs = []
                for m in metadata:
                    try:
                        meta_strs.append(json.dumps(m))
                    except Exception:
                        logger.warning("Failed to serialize metadata; storing {}")
                        meta_strs.append("{}")
                if len(meta_strs) < new_n:
                    meta_strs.extend(["{}"] * (new_n - len(meta_strs)))
                grp.create_dataset(
                    "metadata",
                    data=np.array(meta_strs, dtype=dt),
                    maxshape=(None,),
                    chunks=(chunk_rows,),
                    dtype=dt,
                    compression=self.compression
                )

                # Attributes
                grp.attrs["n_samples"] = new_n
                grp.attrs["embedding_dim"] = buffer['embedding_dim']
                grp.attrs["rank"] = rank
                grp.attrs["compression"] = self.compression
                grp.attrs["source_file"] = source_key

                logger.info(f"Created dataset for source '{source_key}' in {file_path} with {new_n} samples")