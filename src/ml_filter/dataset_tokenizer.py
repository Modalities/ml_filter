from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

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
        annotator_average_fn: Optional[Callable] = np.median,
        annotation_names: Optional[List[str]] = [],
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
        self.annotator_average_fn = annotator_average_fn
        self.annotation_names = annotation_names

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
            annotation_dir_path = Path(annotation_dir_path)
            for i, path in enumerate(file_path.glob("**/*.jsonl")):
                new_dataset = load_dataset(
                    "json",
                    data_files=[str(path)],
                    split=split,
                    cache_dir=cache_dir,
                )
                annotation_paths = self.get_annotation_paths(path, annotation_dir_path, self.annotation_names) * 2
                for annotation_path, prefix in zip(annotation_paths, self.annotation_names):
                    annotation_dataset = load_dataset(
                        "json",
                        data_files=[str(annotation_path)],
                        split=split,
                        cache_dir=cache_dir,
                    )
                    new_dataset = self.join_datasets(
                        new_dataset, annotation_dataset, "id", "document_id", prefix=prefix
                    )
                if i == 0:
                    dataset = new_dataset
                else:
                    dataset = concatenate_datasets([dataset, new_dataset])
            return self._process_dataset_distributed(dataset, self.annotation_names)
        else:
            raise ValueError(f"Invalid path {file_path}. Path must be .jsonl or directory")

    @staticmethod
    def get_annotation_paths(raw_path: Path, annotation_root_dir: Path, annotation_names: List[str]) -> List[Path]:
        raw_suffix = Path(raw_path.parent.parent.name) / Path(raw_path.parent.name)
        raw_filename = raw_path.stem
        annotation_paths = list(annotation_root_dir.joinpath(raw_suffix).glob(f"{raw_filename}*.jsonl"))

        # Sort annotation paths
        def path_sort_key(path: Path) -> List[int]:
            # Find the first occurrence of each model/prompt name in the path string
            indices = [str(path).index(name) for name in annotation_names]
            return indices

        return sorted(annotation_paths, key=path_sort_key)

    @staticmethod
    def join_datasets(
        dataset1: Dataset, dataset2: Dataset, join_column1: str, join_column2: str, prefix: str
    ) -> Dataset:
        """
        Join two Hugging Face datasets on a common column.

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
            match[f"{prefix}_scores"] = match["scores"]

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

    def _process_dataset_distributed(self, dataset: Dataset, annotation_prefixes: List[str]) -> Dataset:
        """Process a dataset by tokenizing text and formatting labels,
        assuming annotations are in distributed in dedicated files."""

        def process_batch(batch):
            # Tokenize the text
            tokenized = self.tokenizer(
                batch[self.text_column], truncation=True, padding=True, max_length=self.max_length, return_tensors="pt"
            )

            # Process labels
            labels = []
            for prefix in annotation_prefixes:
                for annotations in batch[f"{prefix}_{self.label_column}"]:
                    # remove missing annotations
                    annotations = [x for x in annotations if x is not None]
                    # if all annotations are missing, default to 0
                    if annotations == []:
                        annotations = [0]
                    if self.regression:
                        labels.append(self.annotator_average_fn(annotations))
                    else:
                        labels.append(round(self.annotator_average_fn(annotations)))
            labels_reshaped = [
                labels[i * len(annotation_prefixes) : (i + 1) * len(annotation_prefixes)]
                for i in range(len(labels) // len(annotation_prefixes))
            ]

            return {**tokenized, "labels": labels_reshaped}

        return dataset.map(process_batch, batched=True, remove_columns=dataset.column_names)
