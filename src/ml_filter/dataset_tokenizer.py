from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from datasets import Dataset, concatenate_datasets, load_dataset
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
        cache_dir: Optional[str] = None,
        annotation_dir_path: Optional[Union[str, Path]] = None,
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
        file_path = Path(file_path)
        if file_path.suffix == ".jsonl":
            # Load the raw dataset
            dataset = load_dataset(
                "json",
                data_files=[str(file_path)],
                split=split,
                cache_dir=cache_dir,
            )
            return self._process_dataset(dataset)
        elif file_path.is_dir():
            annotation_dir_path = Path(
                "/raid/s3/opengptx/eurolingua/cc_debug_datasets/cc_debug_subset_100_docs_annotations"
            )
            # data_files = []
            for i, path in enumerate(file_path.glob("**/*.jsonl")):
                # data_files.append(str(path))
                new_dataset = load_dataset(
                    "json",
                    data_files=[str(path)],
                    split=split,
                    cache_dir=cache_dir,
                )
                annotation_path = self.get_annotation_path(path, annotation_dir_path)
                annotation_dataset = load_dataset(
                    "json",
                    data_files=[str(annotation_path)],
                    split=split,
                    cache_dir=cache_dir,
                )
                merged_dataset = self.join_datasets(new_dataset, annotation_dataset, "id", "document_id")
                if i == 0:
                    dataset = merged_dataset
                else:
                    dataset = concatenate_datasets([dataset, merged_dataset])

            return self._process_dataset2(dataset)
        else:
            raise ValueError(f"Invalid path {file_path}. Path must be .jsonl or directory")

    @staticmethod
    def get_annotation_path(raw_path: Path, annotation_root_dir: Path) -> Path:
        raw_suffix = "/".join(str(raw_path).split("/")[-3:-1])
        raw_filename = str(raw_path).split("/")[-1].split(".")[0]
        annotation_path = list(Path(str(annotation_root_dir) + "/" + raw_suffix).glob(f"{raw_filename}*.jsonl"))[0]
        return annotation_path

    @staticmethod
    def join_datasets(dataset1: Dataset, dataset2: Dataset, join_column1: str, join_column2: str) -> Dataset:
        """
        Join two Hugging Face datasets on a common column without using pandas.

        Args:
            dataset1 (Dataset): First Hugging Face dataset
            dataset2 (Dataset): Second Hugging Face dataset
            join_column (str): Name of the column to join on

        Returns:
            Dataset: Merged Hugging Face dataset
        """
        # Create a mapping from the join column to rows in dataset2
        dataset2_map = {row[join_column2]: row for row in dataset2}

        # Function to merge rows
        def merge_rows(row: Dict[str, Any]) -> Dict[str, Any]:
            # Find the matching row in dataset2
            match = dataset2_map.get(row[join_column1], {})

            # Merge the dictionaries, with dataset1 rows taking precedence
            merged_row = {**match, **row}
            return merged_row

        # Apply the merge to dataset1
        merged_dataset = dataset1.map(merge_rows)

        return merged_dataset

    def _process_dataset(self, dataset: Dataset) -> Dataset:
        """Process a dataset by tokenizing text and formatting labels."""

        def process_batch(batch):
            # Tokenize the text
            tokenized = self.tokenizer(
                batch[self.text_column], truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
            )

            # Process labels
            labels = []
            for item in batch[self.label_column]:
                if self.regression:
                    labels.append([float(item[k]) for k in self.output_names])
                else:
                    labels.append([int(item[k]) for k in self.output_names])

            return {**tokenized, "labels": labels}

        return dataset.map(process_batch, batched=True, remove_columns=dataset.column_names)

    def _process_dataset2(self, dataset: Dataset) -> Dataset:
        """Process a dataset by tokenizing text and formatting labels."""

        def process_batch(batch):
            # Tokenize the text
            tokenized = self.tokenizer(
                batch[self.text_column], truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
            )

            # Process labels
            labels = []
            for item in batch[self.label_column]:
                item = [x for x in item if x is not None]
                if item == []:
                    item = [0]
                if self.regression:
                    labels.append([np.mean(item)])
                else:
                    labels.append([round(np.median(item))])

            return {**tokenized, "labels": labels}

        return dataset.map(process_batch, batched=True, remove_columns=dataset.column_names)
