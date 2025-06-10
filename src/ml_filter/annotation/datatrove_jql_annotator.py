from typing import Optional

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
import os
import h5py
import numpy as np


def embedding_adapter(self, embedding: np.ndarray, path: str, idx: int) -> dict:
    return {
        "text": "",  # or some dummy placeholder
        "id": f"{path}#{idx}",
        "metadata": {"embedding": embedding.tolist()}
    }


def stats_adapter(writer: DiskWriter, document: Document, expand_metadata=True) -> dict:
    """
    The datatrove adapter to write stats metadata without the actual document text
    """
    data = {key: val for key, val in dataclasses.asdict(document).items() if val and key != "text"}
    if writer.expand_metadata and "metadata" in data:
        data |= data.pop("metadata")
    return data


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

        breakpoint()

        embedder = get_embedder_instance(self.embedder_model_id, device, bfloat16)

        for doc_batch in batched(doc_pipeline, self.batch_size):
            with self.track_time(unit='batch'):
                embeddings = embedder.embed([doc.text for doc in doc_batch])
                for doc, embedding in zip(doc_batch, embeddings):
                    # Convert tensor to list for JSON serialization
                    doc.metadata['embedding'] = embedding.cpu().tolist()
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
            file_progress: bool = False,
            doc_progress: bool = False,
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
                # Need to read file into memory or temp path for h5py
                import tempfile

                with tempfile.NamedTemporaryFile(delete=False, suffix=".h5") as tmp_file:
                    tmp_file.write(fs_file.read())
                    tmp_path = tmp_file.name

            dataset_name = "train"
            with h5py.File(tmp_path, "r") as f:
                if dataset_name not in f:
                    raise KeyError(f"Dataset '{dataset_name}' not found in {filepath}")

                grp = f[dataset_name]
                embeddings = torch.from_numpy(grp["embeddings"][:]).float()
                labels = torch.from_numpy(grp["labels"][:]).float()

                n_samples = grp.attrs["n_samples"]

                logger.info(f"Loaded {n_samples} embeddings from {filepath}:{self.dataset_name}")

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
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
