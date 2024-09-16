import os
from typing import Dict

import torch.nn as nn
from datasets import Dataset
from transformers import Trainer, TrainingArguments

from ml_filter.tokenizer.tokenizer_wrapper import TokenizerWrapper


class TrainingLoop:
    def __init__(self, tokenizer: TokenizerWrapper, sample_key: str):
        self.tokenizer = tokenizer
        self.sample_key = sample_key

    def _tokenize(self, documents: Dict[str, str]):
        return self.tokenizer.tokenizer(
            documents[self.sample_key],
            truncation=self.tokenizer.truncation,
            padding=self.tokenizer.padding,
            max_length=self.tokenizer.max_length,
        )

    def train(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        val_dataset: Dataset,
        batch_size: int,
        weight_decay: float,
        epochs: int,
        eval_strategy: str,
        save_strategy: str,
        logging_steps: int,
        output_dir: str,
        logging_dir: str,
        use_bf16: bool,
        tokenizer: TokenizerWrapper,
        # collator: DataCollator,
    ):
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy=eval_strategy,
            per_device_train_batch_size=batch_size,
            weight_decay=weight_decay,
            num_train_epochs=epochs,
            save_strategy=save_strategy,
            logging_steps=logging_steps,
            logging_dir=logging_dir,
            bf16=use_bf16,
            # TODO: check
            greater_is_better=True,
        )

        # tokenized_dataset = train_dataset.map(self._tokenize, batched=True)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            # data_collator=DataCollatorWithPadding(self.tokenizer),
        )

        trainer.train()
        trainer.save_model(os.path.join(output_dir, "final"))
