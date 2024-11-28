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


class ClassifierTrainingPipeline:
    def __init__(self, config_file_path: Path):
        cfg = OmegaConf.load(config_file_path)

        # Set seeds before loading the model etc.
        self.seed = cfg.training.seed if "seed" in cfg.training else None  # default seed
        if self.seed is not None:
            self._set_seeds()

        # Data
        self.train_data_file_path = cfg.data.train_file_path
        self.train_data_split = cfg.data.train_file_split
        self.val_data_file_path = cfg.data.val_file_path
        self.val_data_split = cfg.data.val_file_split
        self.gt_data_file_path = cfg.data.gt_file_path
        self.gt_data_split = cfg.data.gt_file_split

        # Model
        if isinstance(cfg.model.name, str) and "xlm-roberta" in cfg.model.name.lower():
            self.model = XLMRobertaForMultiTargetClassification.from_pretrained(
                cfg.model.name,
                num_metrics=cfg.data.num_metrics,
                num_classes_per_metric=torch.tensor(cfg.data.num_classes_per_metric),
                regression=cfg.training.regression_loss
            )
        else:
            # Fall back to current initialization for other model types
            self.model = AutoModelForSequenceClassification.from_pretrained(
                cfg.model.name,
                num_labels=cfg.model.num_labels,
                classifier_dropout=cfg.model.classifier_dropout,
                hidden_dropout_prob=cfg.model.hidden_dropout_prob,
                output_hidden_states=cfg.model.output_hidden_states,
            )
        # loss function
        self.regression_loss = cfg.training.regression_loss

        # multilabel settings
        self.num_metrics = cfg.data.num_metrics

        if self.num_metrics > 1:
            self.num_classes_per_metric = torch.tensor(cfg.data.num_classes_per_metric)
            self.metric_names = cfg.data.metric_names
        elif self.num_metrics == 1:
            self.num_classes_per_metric = torch.tensor(cfg.model.num_labels).unsqueeze(0)

        if isinstance(self.model, BertForSequenceClassification):
            self.embedding_size = self.model.classifier.in_features
            if self.regression_loss:
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.num_metrics, bias=True),
                    RegressionScalingLayer(self.num_classes_per_metric),
                )
            else:
                self.model.num_labels = self.num_metrics * max(self.num_classes_per_metric)
                self.model.classifier = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.model.num_labels, bias=True),
                    LogitMaskLayer(self.num_classes_per_metric),
                )
        elif isinstance(self.model, XLMRobertaForSequenceClassification) or isinstance(
            self.model, RobertaForSequenceClassification
        ):
            self.embedding_size = self.model.classifier.dense.in_features
            if self.regression_loss:
                self.model.classifier.out_proj = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.num_metrics, bias=True),
                    RegressionScalingLayer(self.num_classes_per_metric),
                )
            else:
                self.model.num_labels = self.num_metrics * max(self.num_classes_per_metric)
                self.model.classifier.out_proj = torch.nn.Sequential(
                    torch.nn.Linear(self.embedding_size, self.model.num_labels, bias=True),
                    LogitMaskLayer(self.num_classes_per_metric),
                )
        else:
            raise NotImplementedError(f"Unsupported model type {type(self.model)}")

        # Tokenizer
        self.tokenizer = PreTrainedHFTokenizer(
            pretrained_model_name_or_path=cfg.tokenizer.pretrained_model_name_or_path,
            truncation=cfg.tokenizer.truncation,
            padding=cfg.tokenizer.padding,
            max_length=cfg.tokenizer.max_length,
        )
        # Training
        self.batch_size = cfg.training.batch_size
        self.epochs = cfg.training.epochs
        self.use_bf16 = cfg.training.use_bf16
        self.weight_decay = cfg.training.weight_decay
        self.eval_strategy = cfg.training.eval_strategy
        self.save_strategy = cfg.training.save_strategy
        self.output_dir = cfg.training.output_dir_path
        self.greater_is_better = cfg.training.greater_is_better
        self.metric_for_best_model = cfg.training.metric_for_best_model

        self.sample_key = cfg.data.text_column
        self.sample_label = cfg.data.label_column
        self.logging_steps = cfg.training.logging_steps
        self.logging_dir = cfg.training.logging_dir_path

    def _tokenize(self, documents: Dict[str, List[str]]):
        return self.tokenizer.tokenizer(
            documents[self.sample_key],
            truncation=self.tokenizer.truncation,
            padding=self.tokenizer.padding,
            max_length=self.tokenizer.max_length,
        )

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

    def _load_dataset(self, file_path: Path, split: str = "train") -> Dataset:
        return load_dataset("json", data_files=[file_path], split=split)

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
            load_best_model_at_end=True,
            metric_for_best_model=self.metric_for_best_model,
            bf16=self.use_bf16,
            greater_is_better=self.greater_is_better,
        )

    def _map_dataset(self, dataset: Dataset) -> Dataset:
        # Map both tokenization and label assignment
        def process_batch(batch):
            tokenized = self._tokenize(batch)
            if self.num_metrics > 1:
                labels = []
                for item in batch[self.sample_label]:
                    if self.regression_loss:
                        labels.append([float(item[k]) for k in self.metric_names])
                    else:
                        labels.append([int(item[k]) for k in self.metric_names])
            elif self.regression_loss:
                labels = [float(x) for x in batch[self.sample_label]]
            else:
                labels = [int(x) for x in batch[self.sample_label]]

            return {**tokenized, "labels": labels}

        return dataset.map(process_batch, batched=True)

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
            target.view(-1, self.num_metrics),
        )

    def multi_target_mse_loss(
        self,
        input: SequenceClassifierOutput,
        target: Tensor,
        num_items_in_batch: int,
        **kwargs,
    ):
        return torch.nn.functional.mse_loss(
            input["logits"],
            target.view(-1, self.num_metrics),
        )

    def compute_metrics(self, eval_pred: EvalPrediction):
        predictions, labels = eval_pred

        # Convert logits to predicted class
        if not self.regression_loss:
            preds = predictions.argmax(axis=1)
            preds_raw = preds
        else:
            preds = np.round(predictions)
            preds_raw = predictions

        if self.num_metrics == 1:
            # Compute classification metrics
            accuracy = accuracy_score(labels, preds)
            f1 = f1_score(labels, preds, average="weighted")

            # Compute regression-like metrics
            mse = mean_squared_error(labels, preds_raw)
            mae = mean_absolute_error(labels, preds_raw)
            return {"accuracy": accuracy, "f1": f1, "mse": mse, "mae": mae}
        else:
            # TODO: implement macro and micro average
            metric_dict = {}
            for i, name in enumerate(self.metric_names):
                accuracy = accuracy_score(labels[:, i], preds[:, i])
                f1 = f1_score(labels[:, i], preds[:, i], average="weighted")

                mse = mean_squared_error(labels[:, i], preds_raw[:, i])
                mae = mean_absolute_error(labels[:, i], preds_raw[:, i])
                metric_dict.update(
                    {f"{name}_accuracy": accuracy, f"{name}_f1": f1, f"{name}_mse": mse, f"{name}_mae": mae}
                )
            return metric_dict

    def train_classifier(self):
        training_arguments = self._create_training_arguments()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        train_dataset = self._load_dataset(self.train_data_file_path, split=self.train_data_split)
        val_dataset = self._load_dataset(self.val_data_file_path, split=self.val_data_split)

        train_dataset = self._map_dataset(train_dataset)
        val_dataset = self._map_dataset(val_dataset)

        eval_datasets = {"val": val_dataset}
        if self.gt_data_file_path:
            gt_dataset = self._load_dataset(self.gt_data_file_path, split=self.gt_data_split)
            gt_dataset = self._map_dataset(gt_dataset)
            eval_datasets["gt"] = gt_dataset

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer.tokenizer)

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
