import os
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from omegaconf import OmegaConf
from torch import Tensor
from transformers import (
    BertForSequenceClassification,
    DataCollatorWithPadding,
    EvalPrediction,
    Trainer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from ml_filter.dataset_tokenizer import DatasetTokenizer
from ml_filter.tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer
from ml_filter.utils.train_classifier import (
    BertForMultiTargetClassification,
    XLMRobertaForMultiTargetClassification,
    XLMRobertaXLForMultiTargetClassification,
    compute_metrics_for_single_output,
)


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
        self._initialize_dataset_tokenizer(cfg)

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

    def _extract_config_from_cfg(self, cfg: Dict):
        # Data
        self.train_data_file_path = cfg.data.train_file_path
        self.train_data_split = cfg.data.train_file_split
        self.val_data_file_path = cfg.data.val_file_path
        self.val_data_split = cfg.data.val_file_split
        self.gt_data_file_path = cfg.data.gt_file_path
        self.gt_data_split = cfg.data.gt_file_split

        self.train_annotation_path = cfg.data.train_annotation_path
        self.val_annotation_path = cfg.data.val_annotation_path
        if cfg.data.annotator_average_fn == "median":
            self.annotator_average_fn = np.median
        elif cfg.data.annotator_average_fn in ["average", "mean"]:
            self.annotator_average_fn = np.mean
        else:
            raise ValueError("annotator_average_fn must be one of [median, mean]")

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

    def _initialize_model(self, cfg):
        """Initialize and configure the model based on the provided configuration."""
        model_name = cfg.model.name
        model_args = {
            "num_regressor_outputs": cfg.data.num_regressor_outputs,
            "num_classes_per_output": torch.tensor(cfg.data.num_classes_per_output),
            "regression": cfg.training.regression_loss,
        }

        # Initialize base model
        if isinstance(model_name, str):
            if "xlm-roberta-xl" in model_name.lower():
                self.model = XLMRobertaXLForMultiTargetClassification.from_pretrained(model_name, **model_args)
            if "xlm-roberta" or "xlm-v" in model_name.lower():
                self.model = XLMRobertaForMultiTargetClassification.from_pretrained(model_name, **model_args)
            elif "snowflake-arctic" in model_name.lower():
                self.model = BertForMultiTargetClassification.from_pretrained(model_name, **model_args)
            else:
                raise NotImplementedError(
                    f"Model {model_name} not supported. Only Snowflake-Arctic and XLM-RoBERTa models are currently supported."  # noqa
                )
        else:
            raise ValueError(f"Model name must be a string, got {type(model_name)}")

        # Freeze encoder if specified
        if cfg.model.get("freeze_encoder", False):
            self._freeze_encoder()

    def _initialize_dataset_tokenizer(self, cfg):
        # Tokenizer
        self.tokenizer = PreTrainedHFTokenizer(
            pretrained_model_name_or_path=cfg.tokenizer.pretrained_model_name_or_path,
            truncation=cfg.tokenizer.truncation,
            padding=cfg.tokenizer.padding,
            max_length=cfg.tokenizer.max_length,
            add_generation_prompt=False,
        )

        # Tokenizer for
        self.dataset_tokenizer = DatasetTokenizer(
            tokenizer=self.tokenizer.tokenizer,
            text_column=self.sample_key,
            label_column=self.sample_label,
            output_names=self.output_names,
            max_length=cfg.tokenizer.max_length,
            regression=self.regression_loss,
            annotator_average_fn=self.annotator_average_fn,
        )

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
            metrics = compute_metrics_for_single_output(
                labels=labels[:, i],
                preds=preds[:, i],
                preds_raw=preds_raw[:, i],
                thresholds=list(range(1, self.num_classes_per_output[i])),
            )
            metric_dict.update({f"{output_name}/{metric}": metrics[metric] for metric in metrics})
        return metric_dict

    def _dataset_loading(self):
        train_dataset = self.dataset_tokenizer.load_and_tokenize(
            self.train_data_file_path, split=self.train_data_split, annotation_dir_path=self.train_annotation_path
        )
        print(train_dataset["labels"])

        val_dataset = self.dataset_tokenizer.load_and_tokenize(
            self.val_data_file_path, split=self.val_data_split, annotation_dir_path=self.val_annotation_path
        )

        eval_datasets = {"val": val_dataset}
        if self.gt_data_file_path:
            gt_dataset = self.dataset_tokenizer.load_and_tokenize(self.gt_data_file_path, split=self.gt_data_split)
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
            eval_dataset=eval_datasets,
            data_collator=data_collator,
            compute_loss_func=(
                self.multi_target_mse_loss if self.regression_loss else self.multi_target_cross_entropy_loss
            ),
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
        trainer.save_model(os.path.join(self.output_dir, "final"))
