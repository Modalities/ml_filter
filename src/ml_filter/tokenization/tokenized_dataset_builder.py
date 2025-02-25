import logging
from pathlib import Path
from typing import Any

from datasets import Dataset, load_dataset
from datasets.formatting.formatting import LazyBatch
from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


class TokenizedDatasetBuilder:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        text_column: str,
        label_column: str,
        output_names: list[str],
        max_length: int,
        document_id_column: str = "document_id",
        truncation: bool = True,
        padding: bool = True,
    ):
        """Initializes the dataset tokenizer for text processing.

        Args:
            tokenizer (PreTrainedTokenizer): A Hugging Face tokenizer for processing text.
            text_column (str): The name of the column containing text to tokenize.
            label_column (str): The name of the column containing target labels.
            output_names (list[str]): A list of output names to extract from labels.
            max_length (int): The maximum tokenized sequence length.
            document_id_column (str, optional): Column name for unique document IDs. Defaults to "id".
            truncation (bool, optional): Whether to truncate sequences exceeding `max_length`. Defaults to True.
            padding (bool, optional): Whether to pad sequences shorter than `max_length`. Defaults to True.

        Raises:
            Warning: If the tokenizer does not have a padding token, a warning is logged and `eos_token`
              is used instead.

        """

        self.tokenizer = tokenizer
        self.text_column = text_column
        self.label_column = label_column
        self.output_names = output_names
        self.max_length = max_length
        self.document_id_column = document_id_column
        self.truncation = truncation
        self.padding = padding

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
        """Load and tokenizes a dataset from a JSONL file.

        Args:
            file_path (Path): Path to a JSONL file.
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
            logger.info(f"Loading dataset from file: {file_path}")
            dataset = load_dataset(
                "json",
                data_files=str(file_path),
                split=split,
                cache_dir=str(cache_dir) if cache_dir else None,
            )
            logger.info("Dataset loaded successfully. Tokenize dataset...")

            return self._tokenize_dataset(dataset)

        # Invalid case
        else:
            raise ValueError(f"Invalid file path: {file_path}. Expected a JSONL file or a dataset directory.")

    def _tokenize_dataset(self, dataset: Dataset) -> Dataset:
        """Tokenizes text in a Hugging Face dataset."""

        def tokenize_batch(batch: LazyBatch) -> dict[str, Any]:
            # Tokenize the text
            tokenized = self.tokenizer(
                batch[self.text_column],
                truncation=self.truncation,
                padding=self.padding,
                max_length=self.max_length,
                return_tensors="pt",
            )

            return {**tokenized}

        return dataset.map(tokenize_batch, batched=True, remove_columns=dataset.column_names)
