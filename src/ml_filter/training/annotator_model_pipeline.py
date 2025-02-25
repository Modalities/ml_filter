import logging
import os
from functools import partial
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoConfig, AutoModel, DataCollatorWithPadding, EvalPrediction, Trainer, TrainingArguments

from ml_filter.evaluation.evaluate_classifier import compute_metrics_for_single_output
from ml_filter.models.annotator_models import AnnotatorModel, MultiTargetRegressionHead
from ml_filter.tokenization.tokenized_dataset_builder import TokenizedDatasetBuilder
from ml_filter.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer

logger = logging.getLogger(__name__)


def run_annotator_pipeline(config_file_path: Path):
    """Runs the entire classifier training pipeline using AnnotatorModel."""
    logger.info(f"Loading configuration from {config_file_path}")
    try:
        cfg = OmegaConf.load(config_file_path)
        seed = cfg.training.get("seed", None)
        _set_seeds(seed)

        # Initialize components
        tokenizer = _init_tokenizer(cfg=cfg)
        model = _init_model(cfg=cfg)
        tokenized_dataset_builder = _init_tokenized_dataset_builder(cfg=cfg, tokenizer=tokenizer)
        training_args = _init_training_args(cfg=cfg)

        # Load datasets
        train_dataset, eval_datasets, data_collator = _load_datasets(
            cfg=cfg,
            tokenized_dataset_builder=tokenized_dataset_builder,
        )

        # Train classifier
        _train_annotator_model(
            model=model,
            training_args=training_args,
            train_dataset=train_dataset,
            eval_datasets=eval_datasets,
            data_collator=data_collator,
            compute_metrics_fn=_get_compute_metrcis_fn(cfg),
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
    tokenizer = PreTrainedHFTokenizer(cfg.tokenizer.pretrained_model_name_or_path)
    return tokenizer


def _init_model(cfg) -> AnnotatorModel:
    """Initializes the model."""
    logger.info("Initializing model.")
    model_config = AutoConfig.from_pretrained(cfg.model.name)
    model = AnnotatorModel(
        base_model=AutoModel.from_pretrained(pretrained_model_name_or_path=cfg.model.name),
        head=MultiTargetRegressionHead(
            input_dim=model_config.hidden_size,
            num_prediction_tasks=cfg.data.num_tasks,
            num_targets_per_prediction_task=torch.tensor(cfg.data.num_targets_per_task),
        ),
    )
    return model


def _init_tokenized_dataset_builder(cfg, tokenizer) -> TokenizedDatasetBuilder:
    """Initializes the dataset tokenizer."""
    logger.info("Initializing dataset tokenizer.")
    tokenized_dataset_builder = TokenizedDatasetBuilder(
        tokenizer=tokenizer.tokenizer,
        text_column=cfg.data.text_column,
        document_id_column=cfg.data.document_id_column,
        max_length=cfg.tokenizer.get("max_length"),
        padding=cfg.tokenizer.get("padding"),
        truncation=cfg.tokenizer.get("truncation"),
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
    tokenized_dataset_builder: TokenizedDatasetBuilder,
) -> tuple[torch.utils.data.Dataset, dict[str, torch.utils.data.Dataset], DataCollatorWithPadding]:
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
        file_path=Path(cfg.data.train_file_path),
        split=cfg.data.train_file_split,
    )

    val_dataset = tokenized_dataset_builder.load_and_tokenize(
        Path(cfg.data.val_file_path),
        split=cfg.data.val_file_split,
    )

    eval_datasets = {"val": val_dataset}
    if cfg.data.test_file_path:
        test_dataset = tokenized_dataset_builder.load_and_tokenize(
            file_path=Path(cfg.data.test_file_path),
            split=cfg.data.gt_file_split,
        )
        eval_datasets["test"] = test_dataset

    # TODO: Check case when tokenized_dataset_builder.padding = False
    data_collator = DataCollatorWithPadding(tokenizer=tokenized_dataset_builder.tokenizer)

    return train_dataset, eval_datasets, data_collator


def _get_compute_metrcis_fn(cfg: DictConfig) -> partial:
    """Returns a partial function for computing metrics."""
    # Create a partial function with pre-configured arguments
    return partial(
        compute_metrics,
        regression_loss=cfg.training.regression_loss,
        output_names=cfg.data.output_names,
        num_targets_per_task=cfg.data.num_targets_per_task,
    )


def _train_annotator_model(
    model: AnnotatorModel,
    training_args: TrainingArguments,
    train_dataset: torch.utils.data.Dataset,
    eval_datasets: dict[str, torch.utils.data.Dataset],
    data_collator: DataCollatorWithPadding,
    compute_metrics_fn: partial,
) -> None:
    """Trains the annotator model.

    Args:
        model (AnnotatorModel): Model instance
        training_args (TrainingArguments): Hugging Face training arguments
        train_dataset (Dataset): Training dataset
        eval_datasets (Dict[str, Dataset]): Validation and test datasets
        data_collator (DataCollatorWithPadding): Data collator
    """
    logger.info("Initializing Trainer and starting training...")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_datasets,
        data_collator=data_collator,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "final"))

    logger.info("Training complete. Model saved.")


def compute_metrics(
    eval_pred: EvalPrediction,
    regression_loss: bool,
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
        regression_loss (bool): Whether the task is a regression or classification task.
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
    preds = np.round(predictions) if regression_loss else predictions.argmax(axis=1)
    # TODO: Check
    preds_raw = predictions if regression_loss else preds

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
