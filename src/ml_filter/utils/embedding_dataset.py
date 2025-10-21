from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


class EmbeddingDataset(Dataset):
    """Dataset that loads pre-computed embeddings from HDF5 files."""

    def __init__(
        self,
        hdf5_path: Path,
        dataset_name: str,
        *,
        embeddings_dataset: str = "embeddings",
        labels_dataset: str = "labels",
    ):
        self.hdf5_path = Path(hdf5_path)
        self.dataset_name = dataset_name
        self.embeddings_dataset = embeddings_dataset
        self.labels_dataset = labels_dataset

        # Load embeddings and labels into memory
        with h5py.File(self.hdf5_path, "r") as f:
            if dataset_name not in f:
                raise KeyError(f"Dataset '{dataset_name}' not found in {self.hdf5_path}")

            grp = f[dataset_name]

            if self.embeddings_dataset not in grp:
                raise KeyError(
                    f"Embeddings dataset '{self.embeddings_dataset}' not found in {self.hdf5_path}:{dataset_name}"
                )
            if self.labels_dataset not in grp:
                raise KeyError(f"Labels dataset '{self.labels_dataset}' not found in {self.hdf5_path}:{dataset_name}")

            self.embeddings = torch.from_numpy(grp[self.embeddings_dataset][:]).float()
            labels_tensor = torch.from_numpy(grp[self.labels_dataset][:])
            self.labels = labels_tensor.float() if labels_tensor.dtype != torch.float32 else labels_tensor

            self.n_samples = len(self.embeddings)
            self.embedding_dim = self.embeddings.shape[1]
            self.n_tasks = self.labels.shape[1] if len(self.labels.shape) > 1 else 1

        print(f"Loaded {self.n_samples} embeddings from {hdf5_path}:{dataset_name}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {"embeddings": self.embeddings[idx], "labels": self.labels[idx]}
