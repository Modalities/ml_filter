import os
from functools import partial
from pathlib import Path

import h5py
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import ConcatDataset
from transformers import EvalPrediction, SchedulerType, Trainer, TrainingArguments
from transformers.modeling_outputs import SequenceClassifierOutput

from ml_filter.evaluation.evaluate_classifier import compute_metrics_for_single_output
from ml_filter.logger import setup_logging
from ml_filter.models.embedding_model import EmbeddingRegressionConfig, EmbeddingRegressionModel
from ml_filter.training.callbacks import SpearmanEarlyStoppingCallback
from ml_filter.utils.embedding_dataset import EmbeddingDataset
from ml_filter.utils.logging import EvaluationSplitLoggerCallback, SuppressTransformersFLOPWarning

logger = setup_logging()
SuppressTransformersFLOPWarning.install(logger)


def run_embedding_head_training_pipeline(config_file_path: Path):
    """
    Run the embedding-based training pipeline.

    This function orchestrates the process of loading configuration, setting random seeds,
    loading embedding datasets, initializing the embedding-based model and training arguments,
    optionally initializing the regression head weights, and training the model head with early stopping.

    Args:
        config_file_path (Path): Path to the configuration file (YAML or similar).
            The configuration should specify training, data, and model parameters.
    """
    logger.info(f"Loading configuration from {config_file_path}")

    try:
        # Load configuration
        cfg = OmegaConf.load(config_file_path)
        seed = cfg.training.get("seed", None)
        _set_seeds(seed)

        embeddings_dataset = cfg.data.get("embeddings_dataset")
        labels_dataset = cfg.data.get("labels_dataset")

        if embeddings_dataset is None:
            raise ValueError("`data.embeddings_dataset` must be defined in the configuration.")
        if labels_dataset is None:
            raise ValueError("`data.labels_dataset` must be defined in the configuration.")

        # Load embedding datasets
        train_dataset, eval_datasets = _load_embedding_datasets(
            cfg.data.train_file_path,
            cfg.data.val_file_path,
            cfg.data.test_file_path,
            embeddings_dataset=embeddings_dataset,
            labels_dataset=labels_dataset,
        )

        # Create embedding-based model
        model = _init_embedding_model(cfg, embeddings_dataset=embeddings_dataset)

        # Initialize training arguments
        training_args = _init_training_args(cfg)

        # Optionally initialize regression head weights
        if cfg.model.init_regression_weights:
            logger.info("Initializing the regression head weights")
            _init_head_weights(model)

        resolved_metric = _validate_metric_for_best_model(
            eval_datasets=eval_datasets,
            metric_for_best_model=cfg.training.get("metric_for_best_model"),
        )
        if resolved_metric is not None:
            training_args.metric_for_best_model = resolved_metric
        early_stopping_metric = training_args.metric_for_best_model

        # Train model
        _train_model_head(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_datasets=eval_datasets,
            compute_loss_fn=_init_head_loss_fn(cfg),
            compute_metrics_fn=_get_compute_metrics_fn(cfg),
            early_stopping_metric=early_stopping_metric,
        )

        logger.info("Embedding-based training pipeline completed successfully.")

    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}", exc_info=True)
        raise


def find_hdf5_files(directory: Path):
    """Find all HDF5 files in a directory."""
    return list(directory.glob("*.h5")) + list(directory.glob("*.hdf5"))


def load_files(files, dataset_name, split_name, embeddings_dataset, labels_dataset):
    """
    Load multiple HDF5 files for a given dataset split.

    Args:
        files (list): List of file paths or file-like objects pointing to HDF5 files.
        dataset_name (str): Name of the dataset to load from each HDF5 file.
        split_name (str): Name of the data split (e.g., "train", "val", "test") for logging and error handling.
        embeddings_dataset (str): Dataset name inside the HDF5 group that stores embeddings.
        labels_dataset (str): Dataset name inside the HDF5 group that stores labels.

    Returns:
        list: A list of EmbeddingDataset objects, one for each successfully loaded file.
    """

    datasets = []
    for file in files:
        try:
            with h5py.File(file, "r") as f:
                available_datasets = list(f.keys())

            actual_dataset = dataset_name if dataset_name in available_datasets else available_datasets[0]
            if actual_dataset != dataset_name:
                logger.info(f"Dataset '{dataset_name}' not found in {file.name}, using '{actual_dataset}'")

            dataset = EmbeddingDataset(
                file,
                actual_dataset,
                embeddings_dataset=embeddings_dataset,
                labels_dataset=labels_dataset,
            )
            if len(dataset) == 0:
                raise ValueError(f"HDF5 group '{actual_dataset}' in file '{file}' is empty for split '{split_name}'.")
            datasets.append(dataset)
            logger.info(f"✅ Loaded {split_name} file {file.name}: {len(dataset)} samples")

        except Exception as e:
            logger.error(f"Failed to load {split_name} file {file.name}: {e}")
            if split_name == "train":
                raise  # Re-raise for training files
            # Continue for validation/test files
    return datasets


def _load_embedding_datasets(
    training_dir: Path,
    validation_dir: Path,
    test_dir: Path = None,
    dataset_name: str = "train",
    embeddings_dataset: str = "embeddings",
    labels_dataset: str = "labels",
):
    """
    Load and organize training, validation, and optionally test datasets from HDF5 files.

    Args:
        training_dir: Path to directory containing training HDF5 files
        validation_dir: Path to directory containing validation HDF5 files
        test_dir: Optional path to directory containing test HDF5 files
        dataset_name: Name of the dataset within each HDF5 file (default: "data")
        embeddings_dataset: Dataset name inside the HDF5 group that stores embeddings.
        labels_dataset: Dataset name inside the HDF5 group that stores labels.

    Returns:
        tuple: (train_dataset, eval_datasets) where eval_datasets is a dict with 'validation' and optionally 'test'
    """
    # Find files in each directory
    training_files = find_hdf5_files(Path(training_dir))
    validation_files = find_hdf5_files(Path(validation_dir))
    test_files = find_hdf5_files(Path(test_dir)) if test_dir else []

    if not training_files:
        raise ValueError(f"No HDF5 files found in training directory: {training_dir}")

    if not validation_files:
        logger.warning(f"No HDF5 files found in validation directory: {validation_dir}")
    logger.info(f"Found {len(training_files)} training files: {[f.name for f in training_files]}")
    if validation_files:
        logger.info(f"Found {len(validation_files)} validation files: {[f.name for f in validation_files]}")
    if test_dir and test_files:
        logger.info(f"Found {len(test_files)} test files: {[f.name for f in test_files]}")
    elif test_dir:
        logger.warning(f"No HDF5 files found in test directory: {test_dir}")

    # Load and combine training datasets
    train_datasets = load_files(training_files, dataset_name, "train", embeddings_dataset, labels_dataset)
    if not train_datasets:
        raise ValueError("No training datasets could be loaded successfully")

    train_dataset = train_datasets[0] if len(train_datasets) == 1 else ConcatDataset(train_datasets)
    if len(train_datasets) > 1:
        logger.info(f"✅ Combined {len(train_datasets)} training files: {len(train_dataset)} total samples")

    # Load and combine evaluation datasets
    eval_datasets = {}
    for name, files in [("validation", validation_files), ("test", test_files)]:
        if not files:
            continue

        datasets = load_files(files, dataset_name, name, embeddings_dataset, labels_dataset)
        if datasets:
            eval_datasets[name] = datasets[0] if len(datasets) == 1 else ConcatDataset(datasets)
            if len(datasets) > 1:
                logger.info(f"✅ Combined {len(datasets)} {name} files: {len(eval_datasets[name])} total samples")

    # Final summary
    if eval_datasets:
        logger.info(f"📊 Total evaluation datasets loaded: {list(eval_datasets.keys())}")
        for split_name, dataset in eval_datasets.items():
            logger.info("Registered evaluation split '%s' with %d samples.", split_name, len(dataset))
    else:
        logger.warning("No evaluation datasets loaded!")

    return train_dataset, eval_datasets


def _validate_metric_for_best_model(
    eval_datasets: dict[str, torch.utils.data.Dataset],
    metric_for_best_model: str | None,
) -> str | None:
    """
    Ensure the configured metric for selecting the best checkpoint refers to an existing evaluation split.

    Args:
        eval_datasets: Dict of evaluation datasets keyed by split name.
        metric_for_best_model: Configured metric identifier.

    Returns:
        str | None: The metric name to use (identical to `metric_for_best_model` when valid) or None.

    Raises:
        ValueError: If the configured metric references a split that is not available.
    """
    if not metric_for_best_model:
        return None

    metric = metric_for_best_model.strip()
    if not metric.startswith("eval_"):
        raise ValueError(
            "Unsupported value for `metric_for_best_model`: "
            f"'{metric_for_best_model}'. Please provide a full metric name prefixed with 'eval_'."
        )

    remainder = metric[len("eval_") :]
    split_candidate, _, _ = remainder.partition("_")

    if split_candidate not in eval_datasets:
        raise ValueError(
            f"Evaluation split '{split_candidate}' referenced in `metric_for_best_model` "
            f"is not available. Available splits: {list(eval_datasets.keys())}"
        )

    return metric


def _init_embedding_model(cfg: DictConfig, embeddings_dataset: str = "embeddings") -> EmbeddingRegressionModel:
    """Initialize the embedding-based model by reading dimensions from training data."""
    logger.info("Initializing embedding-based model.")

    # Find HDF5 files in training directory
    training_files = find_hdf5_files(Path(cfg.data.train_file_path))
    if not training_files:
        raise ValueError(f"No HDF5 files found in training directory: {cfg.data.train_file_path}")

    # Get embedding dimension from the first training file
    sample_file = training_files[0]
    with h5py.File(sample_file, "r") as f:
        available_datasets = list(f.keys())
        sample_dataset = available_datasets[0]  # Any dataset works for getting dimensions
        if embeddings_dataset not in f[sample_dataset]:
            raise KeyError(
                f"Embeddings dataset '{embeddings_dataset}' not found in {sample_file.name}:{sample_dataset}"
            )
        embedding_dim = f[sample_dataset][embeddings_dataset].shape[1]
        logger.info(f"Reading dimensions from '{sample_file.name}:{sample_dataset}': {embedding_dim}D embeddings")

    config = EmbeddingRegressionConfig(
        embedding_dim=int(embedding_dim),  # Convert numpy.int64 to Python int
        num_tasks=int(cfg.data.num_tasks),  # Convert to Python int
        num_targets_per_task=[int(x) for x in cfg.data.num_targets_per_task],  # Convert each element to Python int
        hidden_dim=int(cfg.model.regressor_hidden_dim),
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
            logger.info("✅ Using Custom regression head initialization")
            logger.info(f"Final layer bias set to {final_layer.bias.data.item()}")


def _init_head_loss_fn(cfg) -> partial:
    """Initialize the loss function for the regression head."""
    num_tasks = cfg.data.num_tasks
    loss_reduction = cfg.training.get("loss_reduction", "mean")
    return partial(mse_loss, num_tasks=num_tasks, reduction=loss_reduction)


def _get_compute_metrics_fn(cfg: DictConfig) -> partial:
    """Get the compute metrics function for the regression head."""
    return partial(
        compute_embedding_metrics,
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
    """
    Train the embedding-based regression model head using Hugging Face's Trainer.

    This function initializes the Trainer with the provided model, training arguments, datasets,
    loss and metrics functions, and early stopping criteria. It performs training and saves the final model.

    Args:
        model (EmbeddingRegressionModel): The embedding-based regression model to be trained.
        training_args (TrainingArguments): Hugging Face training arguments for configuration.
        train_dataset: The dataset used for training. Should be compatible with Hugging Face datasets.
        eval_datasets (dict): A dictionary of evaluation datasets used for validation during training.
        compute_loss_fn (partial): A partially-applied loss computation function.
        compute_metrics_fn (partial): A partially-applied function to compute evaluation metrics.
        early_stopping_metric (str): The name of the evaluation metric to monitor for early stopping.
    """
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
        callbacks=[early_stopping, EvaluationSplitLoggerCallback()],
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
    """
    Computes the mean squared error (MSE) loss for multi-task regression.

    Args:
        input (SequenceClassifierOutput): Model output containing logits of shape (batch_size, num_tasks).
        target (torch.Tensor): Ground truth tensor of shape (batch_size, num_tasks) or (batch_size * num_tasks,).
        num_items_in_batch (int): Number of items in the batch (unused, included for compatibility).
        num_tasks (int): Number of regression tasks (columns in logits/target).
        reduction (str, optional): Specifies the reduction to apply to the output:
            'mean', 'sum', or 'none'. Default is 'mean'.
        ignored_index (int, optional): Specifies a target value that is ignored and does not contribute to the loss.
            Default is -100.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        torch.Tensor: The computed MSE loss (scalar if reduction is 'mean' or 'sum', otherwise per-element).
    """
    logits = input.logits
    target = target.view(-1, num_tasks)
    mask = ~(target == ignored_index)
    target = target.to(dtype=logits.dtype)
    out = torch.pow((logits - target)[mask], 2)
    reduction_normalized = reduction.lower()
    if reduction_normalized == "mean":
        return out.mean()
    elif reduction_normalized == "sum":
        return out.sum()
    elif reduction_normalized == "none":
        return out
    else:
        raise ValueError(f"Unsupported reduction mode '{reduction}'. Expected one of ['mean', 'sum', 'none'].")


def compute_embedding_metrics(
    eval_pred: EvalPrediction,
    task_names: list[str],
    num_targets_per_task: list[int],
) -> dict[str, float]:
    """Computes metrics for multi-task regression heads.

    Args:
        eval_pred (EvalPrediction): A tuple containing predictions of shape
            (batch_size, num_regressor_outputs) and labels of the same shape.
        task_names (list[str]): List of output names for each task.
        num_targets_per_task (list[int]): Number of target classes per task
            (used to derive threshold ranges for binary metrics).

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
    preds = np.round(predictions)
    preds_raw = predictions
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
