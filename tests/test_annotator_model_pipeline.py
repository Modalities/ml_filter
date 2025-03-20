import json
import shutil
from unittest.mock import MagicMock

import numpy as np
import torch
from omegaconf import DictConfig
from transformers import TrainingArguments

from ml_filter.models.annotator_models import AnnotatorModel
from ml_filter.tokenization.tokenized_dataset_builder import DataPreprocessor
from ml_filter.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer
from ml_filter.training.annotator_model_pipeline import (
    _init_model,
    _init_tokenizer,
    _init_training_args,
    _load_datasets,
    _set_seeds,
    run_annotator_training_pipeline,
)


def test_set_seeds():
    _set_seeds(42)
    assert torch.initial_seed() == 42
    assert np.random.get_state()[1][0] == 42


def test_init_tokenizer():
    cfg = DictConfig(
        {
            "tokenizer": {
                "pretrained_model_name_or_path": "bert-base-uncased",
                "add_generation_prompt": False,
            }
        }
    )
    tokenizer = _init_tokenizer(cfg)
    assert isinstance(tokenizer, PreTrainedHFTokenizer)


def test_init_model():
    cfg = DictConfig(
        {
            "model": {"name": "facebookai/xlm-roberta-base", "freeze_base_model_parameters": False},
            "data": {"num_tasks": 3, "num_targets_per_task": [2, 3, 4]},
        }
    )

    model = _init_model(cfg)
    assert isinstance(model, AnnotatorModel)


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
            }
        }
    )
    training_args = _init_training_args(cfg)
    assert isinstance(training_args, TrainingArguments)
    assert training_args.per_device_train_batch_size == 8
    assert training_args.output_dir == "./output"


def test_load_datasets():
    cfg = DictConfig(
        {
            "data": {
                "train_file_path": "train.json",
                "train_file_split": "train",
                "val_file_path": "val.json",
                "val_file_split": "val",
                "test_file_path": "test.json",
                "test_file_split": "test",
            }
        }
    )
    tokenized_dataset_builder = MagicMock(spec=DataPreprocessor)
    tokenized_dataset_builder.load_and_tokenize.side_effect = ["train_ds", "val_ds", "test_ds"]
    train_dataset, eval_datasets = _load_datasets(cfg, tokenized_dataset_builder)
    assert train_dataset == "train_ds"
    assert eval_datasets["val"] == "val_ds"
    assert eval_datasets["test"] == "test_ds"


def test_run_annotator_pipeline(config_file, temp_output_dir):
    """Runs the full pipeline end-to-end."""
    import os

    if os.environ.get('CUDA_VISIBLE_DEVICES') is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # Use only GPUs 0 and 1

    _dummy_dataset_files(temp_output_dir)
    run_annotator_training_pipeline(config_file)

    # Verify model output directory
    model_output_path = temp_output_dir / "final"
    assert model_output_path.exists(), "Model output directory was not created"

    # Check if logs are generated
    logs_path = temp_output_dir / "logs"
    assert logs_path.exists(), "Logs directory was not created"

    # Clean up temp files
    shutil.rmtree(temp_output_dir, ignore_errors=True)


def _dummy_dataset_files(temp_output_dir):
    """Creates dummy dataset files in JSONL format for the test."""
    dummy_data = [
        {"text": "Sample text 1", "labels": [1, 0, 1]},
        {"text": "Sample text 2", "labels": [1, 1, 1]},
        {"text": "Sample text 3", "labels": [0, 0, 0]},
        {"text": "Sample text 4", "labels": [0, 1, 0]},
    ]

    for split in ["train", "val", "test"]:
        dataset_path = temp_output_dir / f"{split}.jsonl"

        # Write each JSON object on a new line
        with dataset_path.open("w") as f:
            for item in dummy_data:
                f.write(json.dumps(item) + "\n")
