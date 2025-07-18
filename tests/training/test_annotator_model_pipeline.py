import random
import shutil
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from transformers import TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from ml_filter.training.embedding_training_pipeline import (
    _init_training_args,
    _set_seeds,
    mse_loss,
    run_embedding_head_training_pipeline,
)


def test_set_seeds():
    _set_seeds(42)
    assert torch.initial_seed() == 42
    assert np.random.get_state()[1][0] == 42

    # Python's built-in random module does not have a direct function to
    # retrieve the currently set seed. Therefore, we test the random sequence.
    random.seed(42)
    expected_sequence = [random.randint(0, 100) for _ in range(5)]
    _set_seeds(42)
    random_numbers = [random.randint(0, 100) for _ in range(5)]
    assert random_numbers == expected_sequence, f"Random sequence mismatch: {random_numbers}"


def test_init_training_args():
    cfg = DictConfig(
        {
            "training": {
                "output_dir_path": "./output",
                "batch_size": 8,
                "epochs": 3,
                "weight_decay": 0.01,
                "learning_rate": 5e-5,
                "save_strategy": "epoch",
                "logging_steps": 10,
                "logging_dir_path": "./logs",
                "metric_for_best_model": "accuracy",
                "use_bf16": False,
                "greater_is_better": False,
                "eval_strategy": "steps",
                "dataloader_num_workers": 4,
                "wandb_run_name": "temp_run",
            }
        }
    )
    training_args = _init_training_args(cfg)
    assert isinstance(training_args, TrainingArguments)
    assert training_args.per_device_train_batch_size == 8
    assert training_args.output_dir == "./output"


# TODO add load dataset test with hdf5 loader


def test_run_embedding_training_pipeline(config_file, temp_output_dir):
    """Runs the full pipeline end-to-end."""
    import os

    if os.environ.get("CUDA_VISIBLE_DEVICES") is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use only GPUs 0 and 1

    _dummy_dataset_files(temp_output_dir)
    run_embedding_head_training_pipeline(config_file)

    # Verify model output directory
    model_output_path = temp_output_dir / "final"
    assert model_output_path.exists(), "Model output directory was not created"

    # Check if logs are generated
    logs_path = temp_output_dir / "logs"
    assert logs_path.exists(), "Logs directory was not created"

    # Clean up temp files
    shutil.rmtree(temp_output_dir, ignore_errors=True)


def test_custom_mse_loss_same_as_torch_implementation():
    num_tasks = 1
    num_items_in_batch = 16
    logits = torch.rand((num_items_in_batch, num_tasks)) * 3.0 - 1.0
    target = torch.randint(0, 2, (num_items_in_batch, num_tasks)).float()
    predicted = SequenceClassifierOutput(logits=logits)
    custom_mse_loss = mse_loss(predicted, target, num_items_in_batch=num_items_in_batch, num_tasks=num_tasks)
    logits = predicted.logits
    target = target.to(dtype=logits.dtype)
    mse_loss_torch = torch.nn.functional.mse_loss(logits, target.view(-1, num_tasks), reduction="mean")
    assert torch.allclose(custom_mse_loss, mse_loss_torch, atol=1e-6), "MSE loss does not match PyTorch implementation"


def _dummy_dataset_files(temp_output_dir: Path):
    """
    Create minimal HDF5 files for train/val/test with 'embeddings' and 'scores' datasets,
    each in its own split‑named subdirectory, using a lowercase .h5 extension.
    """
    splits = {"train": 20, "val": 10, "test": 10}

    for split, n_examples in splits.items():
        # 1) create the split directory
        split_dir = temp_output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)

        # 2) write the HDF5 file inside it with a lowercase .h5 extension
        hdf5_path = split_dir / f"{split}.h5"
        with h5py.File(hdf5_path, "w") as f:
            grp = f.create_group("data")

            # embeddings: dummy embedding vectors (n_examples, embedding_dim)
            # Using 768 as a typical embedding dimension
            embedding_dim = 768
            embeddings = np.random.randn(n_examples, embedding_dim).astype(np.float32)
            grp.create_dataset("embeddings", data=embeddings)

            # labels: regression targets, shape=(n_examples, num_tasks)
            # Based on config: num_tasks=3, so create 3 task labels per example
            num_tasks = 3
            labels = np.random.rand(n_examples, num_tasks).astype(np.float32)
            grp.create_dataset("labels", data=labels)

            # id column
            ids = np.arange(n_examples, dtype=np.int64)
            grp.create_dataset("id", data=ids)
