from pathlib import Path

import h5py
import torch
from torch.utils.data import Dataset


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

            self.n_samples = len(self.embeddings)
            self.embedding_dim = self.embeddings.shape[1]
            self.n_tasks = self.labels.shape[1] if len(self.labels.shape) > 1 else 1

        print(f"Loaded {self.n_samples} embeddings from {hdf5_path}:{dataset_name}")

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return {"embeddings": self.embeddings[idx], "labels": self.labels[idx]}
