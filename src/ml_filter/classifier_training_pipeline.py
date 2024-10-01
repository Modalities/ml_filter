import os
from pathlib import Path
from typing import Dict, List

import torch
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from transformers import AutoModelForSequenceClassification, DataCollatorWithPadding, Trainer, TrainingArguments

from ml_filter.tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer


class ClassifierTrainingPipeline:
    def __init__(self, config_file_path: Path):
        cfg = OmegaConf.load(config_file_path)

        # Data
        self.train_data_file_path = cfg.data.train_file_path
        self.val_data_file_path = cfg.data.val_file_path

        # Model
        # TODO: Check, whetehr AutoModelForSequenceClassification is general enough
        self.model = AutoModelForSequenceClassification.from_pretrained(
            cfg.model.name,
            num_labels=cfg.model.num_labels,
            classifier_dropout=cfg.model.classifier_dropout,
            hidden_dropout_prob=cfg.model.hidden_dropout_prob,
            output_hidden_states=cfg.model.output_hidden_states,
        )

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
            bf16=self.use_bf16,
            # TODO: check
            greater_is_better=True,
        )

    def _map_dataset(self, dataset: Dataset) -> Dataset:
        # Map both tokenization and label assignment
        return dataset.map(
            lambda x: {
                **self._tokenize(x),  # tokenize the text
                "labels": torch.tensor([int(score) for score in x[self.sample_label]], dtype=torch.long),
            },
            batched=True,
        )

    def train_classifier(self):
        training_arguments = self._create_training_arguments()

        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # TODO
        train_dataset = self._load_dataset(self.train_data_file_path)
        val_dataset = self._load_dataset(self.val_data_file_path)

        train_dataset = self._map_dataset(train_dataset)
        val_dataset = self._map_dataset(val_dataset)

        data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer.tokenizer)

        trainer = Trainer(
            model=self.model,
            args=training_arguments,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(os.path.join(self.output_dir, "final"))
