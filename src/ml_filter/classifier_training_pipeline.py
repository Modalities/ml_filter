import os
from pathlib import Path
from typing import Dict, List

import numpy as np

import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch import Tensor
from transformers import (
    AutoModelForSequenceClassification,
    BertForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from ml_filter.tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer
from ml_filter.utils.train_classifier import LogitMaskLayer, RegressionScalingLayer
from ml_filter.utils.train_classifier import XLMRobertaForMultiTargetClassification
from ml_filter.dataset_tokenizer import DatasetTokenizer


class ClassifierTrainingPipeline:
    def __init__(self, config_file_path: Path):
        cfg = OmegaConf.load(config_file_path)

        # Set seeds before loading the model etc.
        self.seed = cfg.training.seed if "seed" in cfg.training else None  # default seed
        if self.seed is not None:
            self._set_seeds()

        self._extract_config_from_cfg(cfg)
        
        # Initialize model
        self._initialize_model(cfg)

        # Initialize dataset
        self._dataset_initialization(cfg)

    def _dataset_initialization(self, cfg):
        # Tokenizer
        self.tokenizer = PreTrainedHFTokenizer(
            pretrained_model_name_or_path=cfg.tokenizer.pretrained_model_name_or_path,
            truncation=cfg.tokenizer.truncation,
            padding=cfg.tokenizer.padding,
            max_length=cfg.tokenizer.max_length,
            add_generation_prompt=False
        )

        # Tokenizer for
        self.dataset_tokenizer = DatasetTokenizer(
            tokenizer=self.tokenizer.tokenizer,
            text_column=self.sample_key,
            label_column=self.sample_label,
            output_names=self.output_names,
            max_length=cfg.tokenizer.max_length,
            regression=self.regression_loss
        )

    def _extract_config_from_cfg(self, cfg: Dict):

        # Data
        self.train_data_file_path = cfg.data.train_file_path
        self.train_data_split = cfg.data.train_file_split
        self.val_data_file_path = cfg.data.val_file_path
        self.val_data_split = cfg.data.val_file_split
        self.gt_data_file_path = cfg.data.gt_file_path
        self.gt_data_split = cfg.data.gt_file_split

        # Training
        self.batch_size = cfg.training.batch_size
        self.epochs = cfg.training.epochs
        self.learning_rate = cfg.training.learning_rate
        self.use_bf16 = cfg.training.use_bf16
        self.weight_decay = cfg.training.weight_decay
        self.eval_strategy = cfg.training.eval_strategy
        self.save_strategy = cfg.training.save_strategy
        self.output_dir = cfg.training.output_dir_path
        self.greater_is_better = cfg.training.greater_is_better
        self.metric_for_best_model = cfg.training.metric_for_best_model
        self.load_best_model_at_end = self.save_strategy != "no"

        self.sample_key = cfg.data.text_column
        self.sample_label = cfg.data.label_column
        self.logging_steps = cfg.training.logging_steps
        self.logging_dir = cfg.training.logging_dir_path

        # loss function
        self.regression_loss = cfg.training.regression_loss

        # multilabel settings
        self.num_regressor_outputs = cfg.data.num_regressor_outputs

        self.num_classes_per_output = torch.tensor(cfg.data.num_classes_per_output)
        self.output_names = cfg.data.output_names

    def _freeze_encoder(self):
        """Freezes all encoder parameters, so that only the classifier is trained."""
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False

        # Unfreeze classifier parameters
        if isinstance(self.model, XLMRobertaForSequenceClassification):
            for param in self.model.classifier.parameters():
                param.requires_grad = True
        elif isinstance(self.model, BertForSequenceClassification):
            # For BERT models, unfreeze both classifier and pooler
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            for param in self.model.bert.pooler.parameters():
                param.requires_grad = True
        else:
            raise NotImplementedError(f"Freezing encoder not implemented for model type {type(self.model)}")

    def _set_seeds(self):
        """Set seeds for reproducibility"""
        import random

        import numpy as np

        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)

            # the following are needed for exact reproducibility across GPUs and runs
            # but slow things down. Don't use them in production.
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    def _create_training_arguments(self) -> TrainingArguments:
        return TrainingArguments(
            output_dir=self.output_dir,
            eval_strategy=self.eval_strategy,
            per_device_train_batch_size=self.batch_size,
            weight_decay=self.weight_decay,
            num_train_epochs=self.epochs,
            save_strategy=self.save_strategy,
            logging_steps=self.logging_steps,
            logging_dir=self.logging_dir,
            seed=self.seed if self.seed is not None else 42,  # 42 is the default value in huggingface Trainer
            # Load best model at the end of training to save it after training in a separate directory
            load_best_model_at_end=self.load_best_model_at_end,
            metric_for_best_model=self.metric_for_best_model,
            bf16=self.use_bf16,
            greater_is_better=self.greater_is_better,
            learning_rate=self.learning_rate,
        )

    def multi_target_cross_entropy_loss(
        self,
        input: SequenceClassifierOutput,
        target: Tensor,
        num_items_in_batch: int,
        **kwargs,
    ):
        """
        The `num_items_in_batch` argument is unused, but this exact signature is required by `Trainer`.
        """
        return torch.nn.functional.cross_entropy(
            input["logits"],
            target.view(-1, self.num_regressor_outputs),
        )

    def multi_target_mse_loss(
        self,
        input: SequenceClassifierOutput,
        target: Tensor,
        num_items_in_batch: int,
        **kwargs,
    ):
        """
        The `num_items_in_batch` argument is unused, but this exact signature is required by `Trainer`.
        """
        return torch.nn.functional.mse_loss(
            input["logits"],
            target.view(-1, self.num_regressor_outputs),
        )

    def compute_metrics(self, eval_pred: EvalPrediction) -> Dict[str, float]:
        """
        Compute metrics for multi-target classification or regression.
        Args:
            eval_pred (EvalPrediction): Contains predictions and labels from evaluation.
                predictions: numpy array of shape (batch_size, num_classes, num_regressor_outputs) for classification
                           or (batch_size, num_regressor_outputs) for regression
                labels: numpy array of shape (batch_size, num_regressor_outputs)
        Returns:
            dict: Dictionary containing computed metrics
        """
        predictions, labels = eval_pred

        # Convert logits to predicted class
        if not self.regression_loss:
            preds = predictions.argmax(axis=1)
            preds_raw = preds
        else:
            # accuracy and F1 are not defined for float predictions
            preds = np.round(predictions)
            preds_raw = predictions
            
        metric_dict = {}
        for i, output_name in enumerate(self.output_names):
            metrics = self._compute_metrics_for_single_output(
                labels=labels[:, i],
                preds=preds[:, i],
                preds_raw=preds_raw[:, i],
                thresholds=list(range(1, self.num_classes_per_output[i]))
            )
            metric_dict.update({
                f"{output_name}/{metric}": metrics[metric]
                for metric in metrics
            })
        return metric_dict

    @staticmethod
    def _compute_metrics_for_single_output(labels: np.ndarray, preds: np.ndarray, preds_raw: np.ndarray, thresholds: list) -> dict:
        """
        Computes evaluation metrics for a specific output.

        Args:
            labels (np.ndarray): Ground truth labels of shape (batch_size,)
            preds (np.ndarray): Predicted class indices of shape (batch_size,)
            preds_raw (np.ndarray): Raw predictions (logits or regression values) of shape (batch_size,) for regression and (batch_size, num_classes) for classification.
            thresholds (list): List of thresholds to use for binary metrics

        Returns:
            dict: Dictionary containing the following metrics:
                - accuracy: Overall classification accuracy
                - f1_weighted: F1 score with weighted averaging
                - f1_micro: F1 score with micro averaging
                - f1_macro: F1 score with macro averaging
                - binary_accuracy_t{t}: Binary accuracy for each threshold t
                - binary_f1_weighted_t{t}: Binary F1 weighted for each threshold t
                - binary_f1_micro_t{t}: Binary F1 micro for each threshold t
                - binary_f1_macro_t{t}: Binary F1 macro for each threshold t
                - mse: Mean squared error between raw predictions and labels
                - mae: Mean absolute error between raw predictions and labels
                - f1_class_{c}: F1 score for each individual class c
        """
        metrics = {}
        
        # Compute classification metrics
        metrics["classification/accuracy"] = accuracy_score(labels, preds)
        metrics["classification/f1_weighted"] = f1_score(labels, preds, average="weighted")
        metrics["classification/f1_micro"] = f1_score(labels, preds, average="micro")
        metrics["classification/f1_macro"] = f1_score(labels, preds, average="macro")

        # Calculate binary metrics for different thresholds
        for threshold in thresholds:
            # Convert to binary predictions using threshold
            binary_preds = np.where(preds >= threshold, 1, 0)
            binary_labels = np.where(labels >= threshold, 1, 0)
            
            metrics[f"binary/t{threshold}/accuracy"] = accuracy_score(binary_labels, binary_preds)
            metrics[f"binary/t{threshold}/f1_weighted"] = f1_score(binary_labels, binary_preds, average="weighted")
            metrics[f"binary/t{threshold}/f1_micro"] = f1_score(binary_labels, binary_preds, average="micro")
            metrics[f"binary/t{threshold}/f1_macro"] = f1_score(binary_labels, binary_preds, average="macro")

        # Compute regression-like metrics
        metrics["regression/mse"] = mean_squared_error(labels, preds_raw)
        metrics["regression/mae"] = mean_absolute_error(labels, preds_raw)
        
        # Add f1 scores for each class
        classes = np.unique(labels)
        classes.sort()
        f1_per_class = f1_score(labels, preds, average=None)
        for i, c in enumerate(classes):
            metrics[f"class_f1/f1_class_{c}"] = f1_per_class[i]
        
        return metrics

    def _dataset_loading(self):
        train_dataset = self.dataset_tokenizer.load_and_tokenize(
            self.train_data_file_path,
            split=self.train_data_split
        )
        
        val_dataset = self.dataset_tokenizer.load_and_tokenize(
            self.val_data_file_path,
            split=self.val_data_split
        )

        eval_datasets = {"val": val_dataset}
        if self.gt_data_file_path:
            gt_dataset = self.dataset_tokenizer.load_and_tokenize(
                self.gt_data_file_path,
                split=self.gt_data_split
            )
            eval_datasets["gt"] = gt_dataset

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer.tokenizer)

        return train_dataset, val_dataset, eval_datasets, data_collator
    def train_classifier(self):
        
        training_arguments = self._create_training_arguments()

        # Load datasets
        train_dataset, val_dataset, eval_datasets, data_collator = self._dataset_loading()

        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_loss_func=(
                self.multi_target_mse_loss if self.regression_loss else self.multi_target_cross_entropy_loss
            ),
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(os.path.join(self.output_dir, "final"))

    def _initialize_model(self, cfg):
        """Initialize and configure the model based on the provided configuration."""
        # Initialize base model
        if isinstance(cfg.model.name, str) and "xlm-roberta" in cfg.model.name.lower():
            self.model = XLMRobertaForMultiTargetClassification.from_pretrained(
                cfg.model.name,
                num_regressor_outputs=cfg.data.num_regressor_outputs,
                num_classes_per_output=torch.tensor(cfg.data.num_classes_per_output),
                regression=cfg.training.regression_loss
            )
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model.name,
                num_labels=cfg.model.num_labels,
                classifier_dropout=cfg.model.classifier_dropout,
                hidden_dropout_prob=cfg.model.hidden_dropout_prob,
                output_hidden_states=cfg.model.output_hidden_states,
            )

        # Configure model classifier based on model type
        if isinstance(self.model, BertForSequenceClassification):
            self.embedding_size = self.model.classifier.in_features
            if self.regression_loss:
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.num_regressor_outputs, bias=True),
                    RegressionScalingLayer(self.num_classes_per_output),
                )
            else:
                self.model.num_labels = self.num_regressor_outputs * max(self.num_classes_per_output)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.model.num_labels, bias=True),
                    LogitMaskLayer(self.num_classes_per_output),
                )
        elif isinstance(self.model, (XLMRobertaForSequenceClassification, RobertaForSequenceClassification)):
            self.embedding_size = self.model.classifier.dense.in_features
            if self.regression_loss:
                self.model.classifier.out_proj = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.num_regressor_outputs, bias=True),
                    RegressionScalingLayer(self.num_classes_per_output),
                )
            else:
                self.model.num_labels = self.num_regressor_outputs * max(self.num_classes_per_output)
                self.model.classifier.out_proj = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.model.num_labels, bias=True),
                    LogitMaskLayer(self.num_classes_per_output),
                )
        else:
            raise NotImplementedError(f"Unsupported model type {type(self.model)}")

        # Freeze encoder if specified
        if cfg.model.get("freeze_encoder", False):
            self._freeze_encoder()
