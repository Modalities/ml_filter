from pathlib import Path
from typing import List, Optional, Union

from datasets import Dataset, load_dataset
from transformers import PreTrainedTokenizer

class DatasetTokenizer:
    """Handles loading and tokenizing datasets from JSONL files."""

    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str,
        label_column: str,
        output_names: List[str],
        max_length: int,
        regression: bool = False,
    ):
        """
        Initialize the dataset tokenizer.

        Args:
            tokenizer: HuggingFace tokenizer to use
            text_column: Name of the column containing text to tokenize
            label_column: Name of the column containing labels
            output_names: List of output names to extract from labels
            max_length: Maximum sequence length for tokenization
            regression: Whether to treat labels as regression values
        """
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.output_names = output_names
        self.max_length = max_length
        self.regression = regression

        # Ensure tokenizer has padding token
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def load_and_tokenize(
        self,
        file_path: Union[str, Path],
        split: str = "train",
        cache_dir: Optional[str] = None
    ) -> Dataset:
        """
        Load a JSONL file and tokenize its contents.

        Args:
            file_path: Path to the JSONL file
            split: Dataset split to use
            cache_dir: Optional directory for caching

        Returns:
            Tokenized dataset
        """
        # Load the raw dataset
        dataset = load_dataset(
            "json",
            data_files=[file_path],
            split=split,
            cache_dir=cache_dir
        )
        
        # Apply tokenization and label processing
        return self._process_dataset(dataset)

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a dataset by tokenizing text and formatting labels."""
        
        def process_batch(batch):
            # Tokenize the text
            tokenized = self.tokenizer(
                batch[self.text_column],
                truncation=True,
                padding=True,
                max_length=self.max_length,
                return_tensors="pt"
            )

            # Process labels
            labels = []
            for item in batch[self.label_column]:
                if self.regression:
                    labels.append([float(item[k]) for k in self.output_names])
                else:
                    labels.append([int(item[k]) for k in self.output_names])

            return {**tokenized, "labels": labels}

        return dataset.map(
            process_batch,
            batched=True,
            remove_columns=dataset.column_names
        ) 