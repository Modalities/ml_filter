import os
from functools import partial
from pathlib import Path
from typing import Callable

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import EvalPrediction, SchedulerType, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from ml_filter.evaluation.evaluate_classifier import compute_metrics_for_single_output
from ml_filter.logger import setup_logging
from ml_filter.models.annotator_models import AnnotatorConfig, AnnotatorModel
from ml_filter.tokenization.tokenized_dataset_builder import DataPreprocessor
from ml_filter.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer
from ml_filter.training.callbacks import SpearmanEarlyStoppingCallback

logger = setup_logging()


def run_annotator_training_pipeline(config_file_path: Path):
    """Runs the entire classifier training pipeline using AnnotatorModel."""
    logger.info(f"Loading configuration from {config_file_path}")
    try:
        cfg = OmegaConf.load(config_file_path)
        seed = cfg.training.get("seed", None)
        _set_seeds(seed)

        # Initialize components
        tokenizer = _init_tokenizer(cfg=cfg)
        tokenized_dataset_builder = _init_tokenized_dataset_builder(cfg=cfg, tokenizer=tokenizer)
        training_args = _init_training_args(cfg=cfg)

        # Load datasets
        train_dataset, eval_datasets = _load_datasets(
            cfg=cfg,
            tokenized_dataset_builder=tokenized_dataset_builder,
        )

        # Initialize model after loading datasets to not use too much memory prematurely
        model = _init_model(cfg=cfg)

        # Train classifier
        _train_annotator_model(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_datasets=eval_datasets,
            compute_loss_fn=_init_loss_fn(cfg=cfg),
            compute_metrics_fn=_get_compute_metrcis_fn(cfg),
            collate=_get_collate_fn(tokenizer=tokenizer),
            tokenizer=tokenizer,
        )

        logger.info("Pipeline execution completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


def _set_seeds(seed: int):
    """Set seeds for reproducibility."""
    if seed is not None:
        logger.info(f"Setting random seed: {seed}")
        import random

        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


def _init_tokenizer(cfg) -> PreTrainedHFTokenizer:
    """Initializes the tokenizer."""
    logger.info("Initializing tokenizer.")
    tokenizer = PreTrainedHFTokenizer(
        pretrained_model_name_or_path=cfg.tokenizer.pretrained_model_name_or_path,
        add_generation_prompt=cfg.tokenizer.add_generation_prompt,
    )
    return tokenizer


def _init_loss_fn(cfg) -> partial:
    num_tasks = cfg.data.num_tasks

    if cfg.training.is_regression:
        return partial(single_target_mse_loss)
    else:
        return partial(multi_target_cross_entropy_loss, num_tasks=num_tasks)


def _init_model(cfg) -> AnnotatorModel:
    """Initializes the model."""
    logger.info("Initializing model.")
    config = AnnotatorConfig(
        is_regression=cfg.model.is_regression,
        num_tasks=cfg.data.num_tasks,
        num_targets_per_task=cfg.data.num_targets_per_task,
        base_model_name_or_path=cfg.model.name,
        load_base_model_from_config=cfg.model.get("load_base_model_from_config", False),
    )
    model = AnnotatorModel(config=config)
    model.set_freeze_base_model(cfg.model.freeze_base_model_parameters)
    return model


def _init_tokenized_dataset_builder(cfg, tokenizer) -> DataPreprocessor:
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


def _init_training_args(cfg) -> TrainingArguments:
    """Initializes the training arguments from the configuration.

    Args:
        cfg: Configuration object containing training parameters.

    Returns:
        TrainingArguments: Hugging Face TrainingArguments object.
    """
    logger.info("Initializing training arguments.")

    training_args = TrainingArguments(
        output_dir=cfg.training.output_dir_path,
        per_device_train_batch_size=cfg.training.batch_size,
        # Default to train batch size
        per_device_eval_batch_size=cfg.training.get("eval_batch_size", cfg.training.batch_size),
        num_train_epochs=cfg.training.epochs,
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.learning_rate,
        # optim = "adamw_hf",
        # max_grad_norm = 2.0,
        # warmup_ratio=0.1,
        lr_scheduler_type=SchedulerType.COSINE,
        save_strategy=cfg.training.save_strategy,
        logging_steps=cfg.training.logging_steps,
        logging_dir=cfg.training.logging_dir_path,
        # Default seed: 42
        seed=cfg.training.get("seed", 42),
        load_best_model_at_end=cfg.training.get("load_best_model_at_end"),
        metric_for_best_model=cfg.training.metric_for_best_model,
        bf16=cfg.training.use_bf16,
        greater_is_better=cfg.training.greater_is_better,
        eval_strategy=cfg.training.eval_strategy,
        # Speed up data loading
        dataloader_num_workers=cfg.training.get("dataloader_num_workers", 4),
    )

    return training_args


def _load_datasets(
    cfg: DictConfig,
    tokenized_dataset_builder: DataPreprocessor,
) -> tuple[torch.utils.data.Dataset, dict[str, torch.utils.data.Dataset]]:
    """Loads and tokenizes datasets for training and evaluation.

    Args:

        cfg: Configuration object
        dataset_tokenizer: DatasetTokenizer instance

    Returns:
        - train_dataset (Dataset)
        - eval_datasets (Dict of validation/test datasets)
        - data_collator (DataCollatorWithPadding)
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


def _get_compute_metrcis_fn(cfg: DictConfig) -> partial:
    """Returns a partial function for computing metrics."""
    # Create a partial function with pre-configured arguments
    return partial(
        compute_metrics,
        is_regression=cfg.training.is_regression,
        task_names=cfg.data.task_names,
        num_targets_per_task=cfg.data.num_targets_per_task,
    )


def _get_collate_fn(
    tokenizer: PreTrainedHFTokenizer,
) -> Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]]:
    """Returns a partial function for collating batches."""
    return partial(collate, pad_token=tokenizer.tokenizer.pad_token_id)


def _train_annotator_model(
    model: AnnotatorModel,
    training_args: TrainingArguments,
    train_dataset: torch.utils.data.Dataset,
    eval_datasets: dict[str, torch.utils.data.Dataset],
    compute_loss_fn: partial,
    compute_metrics_fn: partial,
    collate: Callable[[list[dict[str, torch.Tensor]]], dict[str, torch.Tensor]],
    tokenizer: PreTrainedHFTokenizer | None = None,
) -> None:
    """Trains the annotator model.

    Args:
        model (AnnotatorModel): Model instance
        training_args (TrainingArguments): Hugging Face training arguments
        train_dataset (Dataset): Training dataset
        eval_datasets (Dict[str, Dataset]): Validation and test datasets
        compute_loss_fn (partial): Function to compute loss
        compute_metrics_fn (partial): Function to compute metrics in evaluation
        collate (Callable): Function to collate batches
    """
    logger.info("Initializing Trainer and starting training...")

    for name, param in model.named_parameters():
        if param.is_shared():
            print(f"{name}: shared = {param.is_shared()}")

    SpearmanEarlyStoppingCallback(metric_key="spearman_corr", patience=5, min_delta=1e-3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        compute_loss_func=compute_loss_fn,
        compute_metrics=compute_metrics_fn,
        data_collator=collate,
        # callbacks=[early_stopping],
    )

    trainer.train()
    final_dir = os.path.join(training_args.output_dir, "final")
    trainer.save_model(final_dir)
    if tokenizer is not None:
        tokenizer.tokenizer.save_pretrained(final_dir)

    logger.info("Training complete. Model saved.")


def compute_metrics(
    eval_pred: EvalPrediction,
    is_regression: bool,
    task_names: list[str],
    num_targets_per_task: list[int],
) -> dict[str, float]:
    """Computes metrics for multi-target classification or regression.

    Args:
        eval_pred (EvalPrediction): A tuple containing:
            - `predictions` (np.ndarray): Logits of shape
                (batch_size, num_classes, num_regressor_outputs) for classification, or
                (batch_size, num_regressor_outputs) for regression.
            - `labels` (np.ndarray): Ground truth labels of shape (batch_size, num_regressor_outputs).
        is_regression (bool): Whether the task is a regression or classification task.
        output_names (list[str]): List of output names for each target.
        num_targets_per_task (list[int]): Number of target classes per task.

    Returns:
        dict[str, float]: A dictionary containing computed metrics for each output.
    """
    predictions, labels = eval_pred

    # Validate inputs
    if not isinstance(predictions, np.ndarray) or not isinstance(labels, np.ndarray):
        raise ValueError("Expected `predictions` and `labels` to be NumPy arrays.")
    if predictions.shape[0] != labels.shape[0]:
        raise ValueError(f"Shape mismatch: predictions {predictions.shape}, labels {labels.shape}")

    # Convert logits to predictions
    preds = predictions if is_regression else predictions.argmax(axis=1)
    preds_raw = predictions if is_regression else preds

    # Compute metrics for each target
    metric_dict = {
        f"{task_name}/{metric}": value
        for task_name, num_targets in zip(task_names, num_targets_per_task)
        for metric, value in compute_metrics_for_single_output(
            labels=labels[:, task_names.index(task_name)],
            predictions=preds[:, task_names.index(task_name)],
            predictions_raw=preds_raw[:, task_names.index(task_name)],
            thresholds=list(range(1, num_targets)),
        ).items()
    }

    return metric_dict


def multi_target_cross_entropy_loss(
    input: SequenceClassifierOutput,
    target: torch.Tensor,
    # Signature required by Trainer
    num_items_in_batch: int,
    num_tasks: int,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Computes multi-target cross-entropy loss for classification.

    Args:
        input (SequenceClassifierOutput): Model output containing logits.
        target (Tensor): Target labels of shape (batch_size, num_tasks).
        num_items_in_batch (int): Required by Trainer but unused.
        num_tasks (int): Number of classification targets per sample.
        reduction (str, optional): Reduction mode (`"mean"`, `"sum"`, `"none"`). Defaults to `"mean"`.

    Returns:
        Tensor: Computed cross-entropy loss.
    """
    logits = input.logits
    return torch.nn.functional.cross_entropy(logits, target.view(-1, num_tasks), reduction=reduction)


def single_target_mse_loss(
    input: SequenceClassifierOutput,
    target: torch.Tensor,
    # Signature required by Trainer
    num_items_in_batch: int,
    reduction: str = "mean",
    ignored_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """Computes multi-target mean squared error (MSE) loss for regression.

    Args:
        input (SequenceClassifierOutput): Model output containing logits.
        target (Tensor): Target labels of shape (batch_size, num_tasks).
        num_items_in_batch (int): Required by Trainer but unused.
        num_tasks (int): Number of regression targets per sample.
        reduction (str, optional): Reduction mode (`"mean"`, `"sum"`, `"none"`). Defaults to `"mean"`.

    Returns:
        Tensor: Computed MSE loss.
    """
    logits = input.logits
    target = target.to(dtype=logits.dtype)

    # Create mask for non-ignored indices
    mask = target != ignored_index

    # If no valid targets, return zero loss
    if mask.sum() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    # Apply mask and compute MSE loss
    valid_logits = logits[mask]
    valid_targets = target[mask]

    # Use PyTorch's built-in MSE loss
    if reduction.lower() == "mean":
        return F.mse_loss(valid_logits, valid_targets, reduction="mean")
    elif reduction.lower() == "sum":
        return F.mse_loss(valid_logits, valid_targets, reduction="sum")
    elif reduction.lower() == "none":
        return F.mse_loss(valid_logits, valid_targets, reduction="none")
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def multi_target_mse_loss(
    input: SequenceClassifierOutput,
    target: torch.Tensor,
    # Signature required by Trainer
    num_items_in_batch: int,
    num_tasks: int,
    reduction: str = "mean",
    ignored_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    """Computes multi-target mean squared error (MSE) loss for regression.

    Args:
        input (SequenceClassifierOutput): Model output containing logits.
        target (Tensor): Target labels of shape (batch_size, num_tasks).
        num_items_in_batch (int): Required by Trainer but unused.
        num_tasks (int): Number of regression targets per sample.
        reduction (str, optional): Reduction mode (`"mean"`, `"sum"`, `"none"`). Defaults to `"mean"`.

    Returns:
        Tensor: Computed MSE loss.
    """
    logits = input.logits
    target = target.view(-1, num_tasks)
    mask = ~(target == ignored_index)
    target = target.to(dtype=logits.dtype)
    out = torch.pow((logits - target)[mask], 2)
    if reduction.lower() == "mean":
        return out.mean()
    elif reduction.lower() == "sum":
        return out.sum()
    elif reduction.lower() == "none":
        return out


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


def extract_and_save_embeddings(config_file_path: Path, output_dir: Path):
    """Extract embeddings and save to HDF5. Run this once before training."""
    logger.info(f"Extracting embeddings using config: {config_file_path}")

    cfg = OmegaConf.load(config_file_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize components (same as your existing pipeline)
    tokenizer = _init_tokenizer(cfg=cfg)
    tokenized_dataset_builder = _init_tokenized_dataset_builder(cfg=cfg, tokenizer=tokenizer)
    model = _init_model(cfg=cfg)
    model.eval()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load datasets (same as your existing pipeline)
    train_dataset, eval_datasets = _load_datasets(cfg=cfg, tokenized_dataset_builder=tokenized_dataset_builder)

    hdf5_path = output_dir / "embeddings.h5"

    # Extract embeddings for training dataset
    logger.info("Extracting embeddings for training dataset...")
    train_embeddings, train_labels = _extract_embeddings_from_dataset(
        model=model, dataset=train_dataset, tokenizer=tokenizer, device=device
    )
    _save_embeddings_to_hdf5(train_embeddings, train_labels, hdf5_path, "train")
    logger.info(f"✅ Training embeddings saved: {train_embeddings.shape[0]} samples")

    # Extract embeddings for ALL evaluation datasets (FIXED!)
    if eval_datasets:
        logger.info(f"Found {len(eval_datasets)} evaluation datasets: {list(eval_datasets.keys())}")

        for eval_name, eval_dataset in eval_datasets.items():  # Loop through ALL eval datasets
            logger.info(f"Extracting embeddings for {eval_name} dataset...")
            eval_embeddings, eval_labels = _extract_embeddings_from_dataset(
                model=model, dataset=eval_dataset, tokenizer=tokenizer, device=device
            )
            _save_embeddings_to_hdf5(eval_embeddings, eval_labels, hdf5_path, eval_name)
            logger.info(f"✅ {eval_name} embeddings saved: {eval_embeddings.shape[0]} samples")
    else:
        logger.warning("No evaluation dataset found!")

    logger.info(f"✅ All embeddings saved to {hdf5_path}")

    # Print summary of what was saved
    _print_embedding_summary(hdf5_path)


def _extract_embeddings_from_dataset(model, dataset, tokenizer, device, batch_size=32):
    """Extract embeddings from a dataset using your existing model."""
    from torch.utils.data import DataLoader

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

            # Extract embeddings using your model's new method
            embeddings = model.extract_embeddings(
                input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
            )

            all_embeddings.append(embeddings.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())

    return np.vstack(all_embeddings), np.vstack(all_labels)


def _save_embeddings_to_hdf5(embeddings: np.ndarray, labels: np.ndarray, output_path: Path, dataset_name: str):
    """Save embeddings to HDF5 file."""
    with h5py.File(output_path, "a") as f:
        if dataset_name in f:
            del f[dataset_name]

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
    import h5py

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
