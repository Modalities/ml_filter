from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset
from transformers import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from ml_filter.models.annotator_model_head import (
    AnnotatorHead,
    MultiTargetClassificationHead,
    MultiTargetRegressionHead,
)


class EmbeddingDataset(Dataset):
    """Dataset that loads pre-computed embeddings from HDF5 files."""

    def __init__(self, hdf5_path: Path, dataset_name: str):
        self.hdf5_path = Path(hdf5_path)
        self.dataset_name = dataset_name

        # Load embeddings and labels into memory
        with h5py.File(self.hdf5_path, "r") as f:
            if dataset_name not in f:
                raise KeyError(f"Dataset '{dataset_name}' not found in {self.hdf5_path}")

            grp = f[dataset_name]
            self.embeddings = torch.from_numpy(grp["embeddings"][:]).float()
            self.labels = torch.from_numpy(grp["labels"][:]).float()

            self.n_samples = grp.attrs["n_samples"]
            self.embedding_dim = grp.attrs["embedding_dim"]
            self.n_tasks = grp.attrs["n_tasks"]

        print(f"Loaded {self.n_samples} embeddings from {hdf5_path}:{dataset_name}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {"embeddings": self.embeddings[idx], "labels": self.labels[idx]}


class EmbeddingRegressionConfig(PretrainedConfig):
    """Configuration for embedding-based regression model."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_tasks: int = 1,
        num_targets_per_task: list[int] = None,
        hidden_dim: int = 1000,
        is_regression: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_tasks = num_tasks
        self.num_targets_per_task = list(num_targets_per_task) if num_targets_per_task else [1]
        self.hidden_dim = hidden_dim
        self.is_regression = is_regression


class EmbeddingRegressionModel(PreTrainedModel):
    """Simplified model that works with pre-computed embeddings."""

    config_class = EmbeddingRegressionConfig

    def __init__(self, config: EmbeddingRegressionConfig):
        super().__init__(config)
        self.config = config

        # Initialize the classification head
        self.head = self._build_head(config)

    def _build_head(self, config: EmbeddingRegressionConfig) -> AnnotatorHead:
        """Build the regression or classification head using your existing classes."""
        head_cls = MultiTargetRegressionHead if config.is_regression else MultiTargetClassificationHead
        head_params = {
            "input_dim": config.embedding_dim,
            "num_prediction_tasks": config.num_tasks,
            "num_targets_per_prediction_task": torch.tensor(config.num_targets_per_task, dtype=torch.int64),
        }
        if config.is_regression:
            head_params["hidden_dim"] = config.hidden_dim

        return head_cls(**head_params)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass with pre-computed embeddings."""
        logits = self.head(embeddings)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
