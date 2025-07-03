from functools import partial
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from ml_filter.logger import setup_logging
from ml_filter.models.base_model import BaseModel, BaseModelConfig
from ml_filter.tokenization.tokenized_dataset_builder import DataPreprocessor
from ml_filter.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer

logger = setup_logging()


def _init_tokenizer(cfg: DictConfig) -> PreTrainedHFTokenizer:
    """Initializes the tokenizer."""
    logger.info("Initializing tokenizer.")
    tokenizer = PreTrainedHFTokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.pretrained_model_name_or_path,
        add_generation_prompt=cfg.tokenizer.add_generation_prompt,
    )
    return tokenizer


def _init_model(cfg: DictConfig) -> BaseModel:
    """Initializes the model."""
    logger.info("Initializing model.")
    config = BaseModelConfig(
        is_regression=cfg.model.is_regression,
        num_tasks=cfg.data.num_tasks,
        num_targets_per_task=cfg.data.num_targets_per_task,
        base_model_name_or_path=cfg.model.name,
        load_base_model_from_config=cfg.model.get("load_base_model_from_config", False),
        loading_params=cfg.model.get("loading_params", {}),  # Defaults to empty
    )
    model = BaseModel(config=config)
    model.set_freeze_base_model(cfg.model.freeze_base_model_parameters, cfg.model.freeze_pooling_layer_params)
    return model


def _init_tokenized_dataset_builder(cfg: DictConfig, tokenizer: PreTrainedHFTokenizer) -> DataPreprocessor:
    """Initializes the dataset tokenizer."""
    logger.info("Initializing dataset tokenizer.")
    tokenized_dataset_builder = DataPreprocessor(
        tokenizer=tokenizer.tokenizer,
        text_column=cfg.data.text_column,
        label_column=cfg.data.label_column,
        document_id_column=cfg.data.document_id_column,
        max_length=cfg.tokenizer.get("max_length"),
        is_regression=cfg.training.is_regression,
        padding=cfg.tokenizer.get("padding"),
        truncation=cfg.tokenizer.get("truncation"),
        num_processes=cfg.data.num_processes,
    )
    return tokenized_dataset_builder


def _load_datasets(
    cfg: DictConfig,
    tokenized_dataset_builder: DataPreprocessor,
) -> tuple[torch.utils.data.Dataset, dict[str, torch.utils.data.Dataset]]:
    """Loads and tokenizes datasets for training and evaluation.

    Args:

        cfg: Configuration object
        tokenized_dataset_builder: DataPreprocessor instance

    Returns:
        - train_dataset (Dataset)
        - eval_datasets (Dict of validation/test datasets)
    """
    logger.info("Loading datasets...")

    train_dataset = tokenized_dataset_builder.load_and_tokenize(
        file_or_dir_path=Path(cfg.data.train_file_path),
        split=cfg.data.train_file_split,
    )

    val_dataset = tokenized_dataset_builder.load_and_tokenize(
        file_or_dir_path=Path(cfg.data.val_file_path),
        split=cfg.data.val_file_split,
    )

    eval_datasets = {"val": val_dataset}
    if cfg.data.test_file_path:
        test_dataset = tokenized_dataset_builder.load_and_tokenize(
            file_or_dir_path=Path(cfg.data.test_file_path),
            split=cfg.data.test_file_split,
        )
        eval_datasets["test"] = test_dataset

    return train_dataset, eval_datasets


def collate(batch: list[dict[str, torch.Tensor]], pad_token: int) -> dict[str, torch.Tensor]:
    """Collates a batch of data for training.

    Args:
        batch (list[dict[str, torch.Tensor]]): A dictionary containing the batch data.
        pad_token (int): The padding token ID.

    Returns:
        dict: A dictionary containing the collated batch data.
    """
    # Pad sequences to the maximum length in the batch
    input_ids = pad_sequence(
        [torch.tensor(item["input_ids"]) for item in batch], batch_first=True, padding_value=pad_token
    )
    attention_mask = pad_sequence(
        [torch.tensor(item["attention_mask"]) for item in batch], batch_first=True, padding_value=0
    )
    if "token_type_ids" in batch[0]:
        token_type_ids = pad_sequence(
            [torch.tensor(item["token_type_ids"]) for item in batch], batch_first=True, padding_value=0
        )
    else:
        token_type_ids = None
    labels = pad_sequence([torch.tensor(item["labels"]) for item in batch], batch_first=True, padding_value=-100)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels,
    }


def extract_and_save_embeddings(config_file_path: Path):
    """Extracts embeddings from a model and saves them to an HDF5 file."""
    logger.info(f"Extracting embeddings using config: {config_file_path}")

    cfg = OmegaConf.load(config_file_path)

    assert cfg.data.train_file_path, "train_file_path is required"
    assert cfg.model.name, "model name is required"

    output_dir = Path(cfg.embedding.save_path)
    # Ensure the parent directory of the full path exists
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    # Initialize components
    tokenizer = _init_tokenizer(cfg=cfg)
    tokenized_dataset_builder = _init_tokenized_dataset_builder(cfg=cfg, tokenizer=tokenizer)
    model = _init_model(cfg=cfg)
    model.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load datasets
    train_dataset, eval_datasets = _load_datasets(cfg=cfg, tokenized_dataset_builder=tokenized_dataset_builder)

    # hdf5_save_path = cfg.embedding.save_path

    # Extract embeddings for training dataset
    logger.info("Extracting embeddings for training dataset...")
    train_embeddings, train_labels = _extract_embeddings_from_dataset(
        config=cfg,
        model=model,
        dataset=train_dataset,
        tokenizer=tokenizer,
        device=device,
        batch_size=cfg.training.batch_size,
    )
    _save_embeddings_to_hdf5(train_embeddings, train_labels, output_dir, "train")
    logger.info(f"✅ Training embeddings saved: {train_embeddings.shape[0]} samples")

    # Extract embeddings for ALL evaluation datasets
    if eval_datasets:
        logger.info(f"Found {len(eval_datasets)} evaluation datasets: {list(eval_datasets.keys())}")

        for eval_name, eval_dataset in eval_datasets.items():  # Loop through ALL eval datasets
            logger.info(f"Extracting embeddings for {eval_name} dataset...")
            eval_embeddings, eval_labels = _extract_embeddings_from_dataset(
                config=cfg,
                model=model,
                dataset=eval_dataset,
                tokenizer=tokenizer,
                device=device,
                batch_size=cfg.training.eval_batch_size,
            )
            _save_embeddings_to_hdf5(eval_embeddings, eval_labels, output_dir, eval_name)
            logger.info(f"✅ {eval_name} embeddings saved: {eval_embeddings.shape[0]} samples")
    else:
        logger.warning("No evaluation dataset found!")

    logger.info(f"✅ All embeddings saved to {output_dir}")

    # Print summary of what was saved
    _print_embedding_summary(output_dir)


def _extract_embeddings_from_dataset(
    config, model, dataset, tokenizer, device, batch_size
) -> Tuple[np.ndarray, np.ndarray]:
    """Extracts embeddings from a dataset using the provided model and tokenizer."""
    logger.info("Extracting embeddings from dataset...")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate, pad_token=tokenizer.tokenizer.pad_token_id),
    )

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting embeddings"):
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)

            # Extract embeddings
            embeddings = model.extract_embeddings(
                config, input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    return np.vstack(all_embeddings), np.vstack(all_labels)


def _save_embeddings_to_hdf5(embeddings: np.ndarray, labels: np.ndarray, output_path: Path, dataset_name: str):
    """Save embeddings to HDF5 file."""
    with h5py.File(output_path, "a") as f:
        if dataset_name in f:
            raise ValueError(
                f"Dataset '{dataset_name}' already exists in {output_path}. "
                f"Please remove the existing file or use a different output path."
            )

        grp = f.create_group(dataset_name)
        grp.create_dataset("embeddings", data=embeddings, compression="gzip")
        grp.create_dataset("labels", data=labels, compression="gzip")

        # Save metadata
        grp.attrs["n_samples"] = embeddings.shape[0]
        grp.attrs["embedding_dim"] = embeddings.shape[1]
        grp.attrs["n_tasks"] = labels.shape[1] if len(labels.shape) > 1 else 1

    logger.info(f"Saved {embeddings.shape[0]} embeddings to {output_path}:{dataset_name}")


def _print_embedding_summary(hdf5_path: Path):
    """Print a summary of the saved embeddings."""

    with h5py.File(hdf5_path, "r") as f:
        logger.info("\n📊 Embedding Summary:")
        total_samples = 0
        for dataset_name in f.keys():
            grp = f[dataset_name]
            n_samples = grp.attrs["n_samples"]
            embedding_dim = grp.attrs["embedding_dim"]
            n_tasks = grp.attrs["n_tasks"]
            total_samples += n_samples
            logger.info(f"  {dataset_name}: {n_samples:,} samples, {embedding_dim}D embeddings, {n_tasks} tasks")
        logger.info(f"  Total: {total_samples:,} samples across all datasets")
