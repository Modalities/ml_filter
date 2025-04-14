import logging
from glob import glob
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class DataPreprocessor:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str,
        label_column: str,
        max_length: int,
        is_regression: bool,
        num_processes: int,
        document_id_column: str = "document_id",
        truncation: bool = True,
        padding: bool = True,
    ):
        """Initializes the dataset tokenizer for text processing.

        Args:
            tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer for processing text.
            text_column (str): The name of the column containing text to tokenize.
            label_column (str): The name of the column containing labels.
            max_length (int): The maximum tokenized sequence length.
            is_regression (bool): Whether the task is regression or classification.
            document_id_column (str, optional): Column name for unique document IDs. Defaults to "document_id".
            truncation (bool, optional): Whether to truncate sequences exceeding `max_length`. Defaults to True.
            padding (bool, optional): Whether to pad sequences shorter than `max_length`. Defaults to True.
            num_processes (int): Number of processes to use for tokenization.

        Raises:
            Warning: If the tokenizer does not have a padding token, a warning is logged and `eos_token`
              is used instead.

        """

        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.max_length = max_length
        self.is_regression = is_regression
        self.document_id_column = document_id_column
        self.truncation = truncation
        self.padding = padding
        self.num_processes = num_processes

        # Ensure tokenizer has padding token
        if not self.tokenizer.pad_token and self.padding:
            logger.warning(
                "Tokenizer has no padding token. Using eos_token as padding token. "
                "This may affect model performance."
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_and_tokenize(
        self,
        file_path: Path,
        split: str = "train",
        cache_dir: str | None = None,
    ) -> Dataset:
        """Load and tokenizes a dataset from a JSONL file(s).

        Args:
            file_path (Path): Path to a JSONL file or a directory structure containing JSONL files.
            split (str, optional): Dataset split to use (default: "train").
            cache_dir (Optional[str], optional): Directory for caching dataset (default: None).

        Returns:
            Dataset: Tokenized Hugging Face dataset.

        Raises:
            FileNotFoundError: If `file_path` does not exist.
            ValueError: If `file_path` is neither a JSONL file nor a directory.
        """

        # Check if path exists
        if not file_path.exists():
            raise FileNotFoundError(f"File or directory not found: {file_path}")

        # Handle JSONL file case
        if file_path.is_file() and file_path.suffix == ".jsonl":
            dataset = self._load_dataset(file_path, split, cache_dir)
            return self._preprocess_dataset(dataset)
        # Handle dataset directory case
        elif file_path.is_dir() and len(glob(str(path := file_path / "**" / "*.jsonl"), recursive=True)) > 0:
            dataset = self._load_dataset(path, split, cache_dir)
        # Invalid case
        else:
            raise ValueError(f"Invalid file path: {file_path}. Expected a JSONL file or a dataset directory.")

        return self._preprocess_dataset(dataset)

    def _load_dataset(self, file_path: Path, split: str, cache_dir: str | None) -> Dataset:
        logger.info(f"Loading dataset from: {file_path}")
        dataset = load_dataset(
            "json",
            data_files=str(file_path),
            split=split,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
        logger.info("Dataset loaded successfully.")
        return dataset

    def _preprocess_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenizes text in a Hugging Face dataset."""
        logger.info("Tokenizing dataset...")

        def process_batch(batch: LazyBatch) -> dict[str, Any]:
            tokenized = self.tokenizer(
                batch[self.text_column],
                truncation=self.truncation,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors="pt",
            )
            labels = self._take_scores(batch)
            assert len(labels.shape) == 2

            return {**tokenized, "labels": labels}

        return dataset.map(
            process_batch,
            batched=True,
            remove_columns=dataset.column_names,
            num_proc=self.num_processes,
            load_from_cache_file=False,
        )

    def _take_scores(self, batch: LazyBatch) -> torch.Tensor:
        """Extracts scores from a batch."""
        dtype = torch.float if self.is_regression else torch.long
        scores_entry = batch[self.label_column]
        if isinstance(scores_entry, dict):
            scores_entry = self._take_scores_from_sub_dict(scores_entry)
        if isinstance(scores_entry, list):
            if isinstance(scores_entry[0], dict):
                scores_entry = [self._take_scores_from_sub_dict(entry) for entry in scores_entry]
            return torch.tensor(scores_entry, dtype=dtype)
        elif isinstance(scores_entry, (int, float)):
            return torch.tensor([scores_entry], dtype=dtype)
        else:
            raise ValueError(f"Unknown score entry format: {scores_entry}")

    def _take_scores_from_sub_dict(self, score_dict: dict[str, int | float | list[int | float]]) -> list[int | float]:
        """Extracts scores from a sub-dictionary."""
        if self.label_column in score_dict:
            entry = score_dict[self.label_column]
        elif "score" in score_dict:
            entry = score_dict["score"]
        elif "scores" in score_dict:
            entry = score_dict["scores"]
        else:
            raise ValueError(f"Unknown score entry format: {score_dict}")
        if isinstance(entry, list):
            return entry
        elif isinstance(entry, (int, float)):
            return [entry]
        else:
            raise ValueError(f"Unknown score entry format: {score_dict}")
