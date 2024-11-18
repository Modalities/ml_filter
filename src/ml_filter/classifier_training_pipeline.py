import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

from .tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer

sys.path.append(os.path.join(os.getcwd(), "src"))


class ClassifierTrainingPipeline:
    def __init__(self, config_file_path: Path):
        cfg = OmegaConf.load(config_file_path)

        # Data
        self.train_data_file_path = cfg.data.train_file_path
        self.val_data_file_path = cfg.data.val_file_path
        self.label_column = cfg.data.label_column

        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Model
        # TODO: Check, whetehr AutoModelForSequenceClassification is general enough
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name,
            num_labels=cfg.model.num_labels,
            classifier_dropout=cfg.model.classifier_dropout,
            hidden_dropout_prob=cfg.model.hidden_dropout_prob,
            output_hidden_states=cfg.model.output_hidden_states,
        )

        # multilabel settings
        self.num_metrics = cfg.data.num_metrics

        if self.num_metrics > 1:
            self.num_classes_per_metric = torch.tensor(cfg.data.num_classes_per_metric)
            self.max_num_labels_per_metric = max(self.num_classes_per_metric)
            self.model.num_labels = self.num_metrics * self.max_num_labels_per_metric

            self.model.classifier = torch.nn.Linear(768, self.model.num_labels, bias=True)
            self.raw_logit_mask = (
                torch.arange(self.max_num_labels_per_metric).repeat(self.num_metrics, 1).T < self.num_classes_per_metric
            ).to(device)
            self.logit_mask = (self.raw_logit_mask + 1e-45).log()
        else:
            self.max_num_labels_per_metric = cfg.model.num_labels
            self.logit_mask = torch.zeros(self.max_num_labels_per_metric, self.num_metrics).to(device)

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

    def _load_dataset(self, file_path: Path) -> Dataset:
        return load_dataset("json", data_files=[file_path])

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
            # Load best model at the end of training to save it after training in a separate directory
            load_best_model_at_end=True,
            bf16=self.use_bf16,
            greater_is_better=self.greater_is_better,
        )

    def _map_dataset(self, dataset: Dataset) -> Dataset:
        # Map both tokenization and label assignment
        if self.num_metrics > 1:
            keys = sorted(dataset[self.label_column][0].keys())

        def process_batch(batch):
            tokenized = self._tokenize(batch)
            if self.num_metrics > 1:
                labels = []
                for item in batch[self.label_column]:
                    labels.append([item[k] for k in keys])
            else:
                labels = batch[self.label_column]

            return {**tokenized, "labels": labels}

        return dataset.map(process_batch, batched=True)

    def reshape_and_mask_logits(self, logits):
        return logits.view(-1, self.max_num_labels_per_metric, self.num_metrics) + self.logit_mask

    def multi_target_cross_entropy_loss(
        self,
        input,
        target,
        num_items_in_batch,
        **kwargs,
    ):
        return torch.nn.functional.cross_entropy(
            self.reshape_and_mask_logits(input["logits"]),
            target.view(-1, self.num_metrics),
        )

    def train_classifier(self):
        training_arguments = self._create_training_arguments()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # TODO
        train_dataset = self._load_dataset(self.train_data_file_path)["train"]
        val_dataset = self._load_dataset(self.val_data_file_path)["train"]

        train_dataset = self._map_dataset(train_dataset)
        val_dataset = self._map_dataset(val_dataset)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            compute_loss_func=self.multi_target_cross_entropy_loss,
        )

        trainer.train()
        trainer.save_model(os.path.join(self.output_dir, "final"))
