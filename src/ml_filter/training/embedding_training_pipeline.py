import os
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from transformers import EvalPrediction, SchedulerType, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from ml_filter.evaluation.evaluate_classifier import compute_metrics_for_single_output
from ml_filter.logger import setup_logging
from ml_filter.models.embedding_model import EmbeddingRegressionConfig, EmbeddingRegressionModel
from ml_filter.training.callbacks import SpearmanEarlyStoppingCallback
from ml_filter.utils.embedding_dataset import EmbeddingDataset

logger = setup_logging()


def run_embedding_head_training_pipeline(config_file_path: Path):
    """Main function to run the embedding-based training pipeline.
    Args:
        config_file_path (Path): Path to the configuration file.
    """

    logger.info(f"Loading configuration from {config_file_path}")

    try:
        cfg = OmegaConf.load(config_file_path)
        embeddings_hdf5_path = cfg.embedding.load_path
        logger.info(f"Loading the embeddings from {embeddings_hdf5_path}")

        seed = cfg.training.get("seed", None)
        _set_seeds(seed)

        # Load embedding datasets
        train_dataset, eval_datasets = _load_embedding_datasets(embeddings_hdf5_path)

        # Create embedding-based model
        model = _init_embedding_model(cfg, embeddings_hdf5_path)

        # Initialize training arguments
        training_args = _init_training_args(cfg)

        if cfg.embedding.init_regression_weights:
            logger.info("Initializing the regression head weights")
            _init_head_weights(model)

        # Train model
        _train_model_head(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_datasets=eval_datasets,
            compute_loss_fn=_init_head_loss_fn(cfg),
            compute_metrics_fn=_get_compute_metrics_fn(cfg),
            early_stopping_metric=cfg.training.metric_for_best_model,
        )

        logger.info("Embedding-based training pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


def _load_embedding_datasets(embeddings_hdf5_path: Path):
    """Load datasets from HDF5 file - loads train + ALL eval datasets."""
    embeddings_path = Path(embeddings_hdf5_path)

    # Check what datasets are available
    with h5py.File(embeddings_path, "r") as f:
        available_datasets = list(f.keys())
        logger.info(f"Available datasets in HDF5 file: {available_datasets}")

    # Load training dataset (must exist)
    if "train" not in available_datasets:
        raise ValueError(f"Training dataset not found in {embeddings_path}")

    train_dataset = EmbeddingDataset(embeddings_path, "train")
    logger.info(f"✅ Loaded training dataset: {len(train_dataset)} samples")

    # Load ALL evaluation datasets (not just the first one!)
    eval_datasets = {}
    non_train_datasets = [name for name in available_datasets if name != "train"]

    if len(non_train_datasets) > 0:
        for eval_name in non_train_datasets:
            eval_datasets[eval_name] = EmbeddingDataset(embeddings_path, eval_name)
            logger.info(f"✅ Loaded {eval_name} dataset: {len(eval_datasets[eval_name])} samples")

        logger.info(f"📊 Total evaluation datasets loaded: {len(eval_datasets)} ({list(eval_datasets.keys())})")
    else:
        logger.warning("No evaluation dataset found!")

    return train_dataset, eval_datasets


def _init_embedding_model(cfg: DictConfig, embeddings_hdf5_path: Path) -> EmbeddingRegressionModel:
    """Initialize the embedding-based model."""
    logger.info("Initializing embedding-based model.")

    # Get embedding dimension from the HDF5 file
    with h5py.File(embeddings_hdf5_path, "r") as f:
        available_datasets = list(f.keys())
        sample_dataset = available_datasets[0]  # Any dataset works for getting dimensions
        embedding_dim = f[sample_dataset].attrs["embedding_dim"]
        logger.info(f"Reading dimensions from '{sample_dataset}' dataset: {embedding_dim}D embeddings")

    config = EmbeddingRegressionConfig(
        embedding_dim=int(embedding_dim),  # Convert numpy.int64 to Python int
        num_tasks=int(cfg.data.num_tasks),  # Convert to Python int
        num_targets_per_task=[int(x) for x in cfg.data.num_targets_per_task],  # Convert each element to Python int
        hidden_dim=int(cfg.model.regressor_hidden_dim),
        is_regression=bool(cfg.model.is_regression),  # Convert to Python bool
    )

    return EmbeddingRegressionModel(config=config)


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


def _init_training_args(cfg) -> TrainingArguments:
    """Initialize training arguments based on the configuration."""

    logger.info("Initializing training arguments.")

    return TrainingArguments(
        output_dir=cfg.training.output_dir_path,
        per_device_train_batch_size=cfg.training.batch_size,
        per_device_eval_batch_size=cfg.training.get("eval_batch_size", cfg.training.batch_size),
        num_train_epochs=cfg.training.epochs,
        weight_decay=cfg.training.weight_decay,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=SchedulerType.COSINE,
        save_strategy=cfg.training.save_strategy,
        logging_steps=cfg.training.logging_steps,
        logging_dir=cfg.training.logging_dir_path,
        seed=cfg.training.get("seed", 42),
        load_best_model_at_end=cfg.training.get("load_best_model_at_end"),
        metric_for_best_model=cfg.training.metric_for_best_model,
        bf16=cfg.training.use_bf16,
        greater_is_better=cfg.training.greater_is_better,
        eval_strategy=cfg.training.eval_strategy,
        dataloader_num_workers=cfg.training.get("dataloader_num_workers", 4),
        report_to=["wandb"],
        run_name=cfg.training.wandb_run_name,
    )


def _init_head_weights(model: EmbeddingRegressionModel):
    """Initialize the regression head weights."""
    logger.info("Initializing regression head weights.")
    with torch.no_grad():
        if hasattr(model.head, "mlp") and len(model.head.mlp) >= 3:
            final_layer = model.head.mlp[2]  # The Linear(1000 -> 1) layer
            final_layer.weight.data *= 10.0  # Scale up weights
            final_layer.bias.data.fill_(2.5)  # Bias toward middle of [0,5] range
            logger.info("✅ Fixed regression head initialization")
            logger.info(f"Final layer bias set to {final_layer.bias.data.item()}")


def _init_head_loss_fn(cfg) -> partial:
    """Initialize the loss function for the regression or classification head."""
    num_tasks = cfg.data.num_tasks
    if cfg.training.is_regression:
        return partial(mse_loss, num_tasks=num_tasks)
    else:
        num_tasks = cfg.data.num_tasks
        return partial(cross_entropy_loss, num_tasks=num_tasks)


def _get_compute_metrics_fn(cfg: DictConfig) -> partial:
    """Get the compute metrics function for the regression or classification head."""
    return partial(
        compute_embedding_metrics,
        is_regression=cfg.training.is_regression,
        task_names=cfg.data.task_names,
        num_targets_per_task=cfg.data.num_targets_per_task,
    )


def _train_model_head(
    model: EmbeddingRegressionModel,
    training_args: TrainingArguments,
    train_dataset,
    eval_datasets: dict,
    compute_loss_fn: partial,
    compute_metrics_fn: partial,
    early_stopping_metric: str,
) -> None:
    """Train the embedding-based model head."""
    logger.info("Initializing Trainer and starting training...")

    early_stopping = SpearmanEarlyStoppingCallback(metric_key=early_stopping_metric, patience=5, min_delta=1e-3)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        compute_loss_func=compute_loss_fn,
        compute_metrics=compute_metrics_fn,
        data_collator=collate_embeddings,
        callbacks=[early_stopping],
    )

    trainer.train()

    final_dir = os.path.join(training_args.output_dir, "final")
    trainer.save_model(final_dir)
    logger.info("Training complete. Model saved.")


def mse_loss(
    input: SequenceClassifierOutput,
    target: torch.Tensor,
    num_items_in_batch: int,
    num_tasks: int,
    reduction: str = "mean",
    ignored_index: int = -100,
    **kwargs,
) -> torch.Tensor:
    logits = input.logits
    target = target.view(-1, num_tasks)
    mask = ~(target == ignored_index)
    target = target.to(dtype=logits.dtype)
    out = torch.pow((logits - target)[mask], 2)
    reduction = "mean"
    if reduction.lower() == "mean":
        return out.mean()
    elif reduction.lower() == "sum":
        return out.sum()
    elif reduction.lower() == "none":
        return out


def cross_entropy_loss(
    input: SequenceClassifierOutput,
    target: torch.Tensor,
    num_items_in_batch: int,
    num_tasks: int,
    reduction: str = "mean",
    **kwargs,
) -> torch.Tensor:
    """Compute cross-entropy loss for multi-target classification."""
    logits = input.logits
    return F.cross_entropy(logits, target.view(-1, num_tasks), reduction=reduction)


def compute_embedding_metrics(
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
    preds = np.round(predictions) if is_regression else predictions.argmax(axis=1)
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


def collate_embeddings(batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Collate function for embedding datasets."""
    embeddings = torch.stack([item["embeddings"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])

    return {"embeddings": embeddings, "labels": labels}
