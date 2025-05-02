import json
import logging
import os
import random
from pathlib import Path
from typing import Callable

from datasets import load_dataset
from tqdm import tqdm

from ml_filter.analysis.utils import custom_round

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def convert_hf_dataset_to_jsonl(
    output_dir_path: Path,
    hf_dataset_name: str,
    hf_dataset_split: str = "train",
):
    """Converts a Hugging Face dataset into JSONL format compatible with ML Filter Classifier.

    This function downloads a dataset of text ("text" field) that was annotated by LLMs ("score" field)
    from HuggingFace, processes each example by converting
    it into a standardized JSONL format with document ID, text content, and scores. The
    processed data is saved to a JSONL file, and the downloaded dataset cache is cleaned up
    afterwards.

    Note: This script is designed to work with the HuggingFaceFW/fineweb-edu-llama3-annotations dataset.

    Args:
        output_dir_path (str): Directory path where the converted JSONL file will be saved.
            The directory will be created if it doesn't exist.
        output_file_name (str, optional): Name of the output JSONL file.
            E.g., "annotated_fineweb.jsonl".
        dataset_name (str, optional): Name of the Hugging Face dataset to download and convert.
            E.g., "HuggingFaceFW/fineweb-edu-llama3-annotations".

    Returns:
        None

    Example:
        >>> convert_to_jsonl("data", "my_dataset.jsonl", "some/dataset")
        Loading dataset: some/dataset...
        Converting to JSONL format...
        Conversion complete! File saved to data/my_dataset.jsonl
        Cleaning up downloaded data...
        Done!
    """
    # Load the dataset
    logger.info(f"Loading dataset: {hf_dataset_name}...")
    dataset = load_dataset(hf_dataset_name, split=hf_dataset_split)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir_path, exist_ok=True)

    # Open output file
    logger.info("Converting to JSONL format...")
    with open(output_dir_path, "w", encoding="utf-8") as f:
        # Process each example
        for idx, example in tqdm(enumerate(dataset), total=len(dataset), desc="Processing dataset"):
            # Create entry in desired format
            entry = {"id": str(idx), "text": example["text"], "scores": {"score": example["score"]}}

            # Write to file
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    logger.info(f"Conversion complete! File saved to {output_dir_path}")

    # Delete downloaded dataset cache
    logger.info("Cleaning up downloaded data...")
    dataset.cleanup_cache_files()
    logger.info("Done!")


def apply_score_transforms(
    input_file_path: Path,
    output_path: Path,
    transform_fns: list[tuple[str, Callable[[float], int | float]]],
):
    """Transform single scores into multiple scores using different transformations.

    Args:
        base_path: Base directory path for data files
        transform_fns: List of tuples containing (name, transform_function) pairs.
                      The original score will always be kept.

    Raises:
        KeyError: If 'scores' key is missing in the JSONL entry or if 'score' is missing in the scores dict.
    """

    # Always include original score first
    transforms = [("score", lambda x: int(custom_round(x)))] + transform_fns

    logger.info("Applying score transformations...")

    with open(input_file_path, "r", encoding="utf-8") as in_f, open(output_path, "w", encoding="utf-8") as out_f:
        for line in tqdm(in_f, desc="Processing entries"):
            try:
                entry = json.loads(line)

                # Validate required keys
                if "scores" not in entry or "score" not in entry["scores"]:
                    logger.warning(f"Skipping entry due to missing keys: {entry}")
                    continue

                original_score = entry["scores"]["score"]

                # Apply all transformations
                entry["scores"] = {name: transform_fn(original_score) for name, transform_fn in transforms}

                # Write transformed entry
                out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            except json.JSONDecodeError as e:
                logger.error(f"Skipping invalid JSON entry: {e}")
            except Exception as e:
                logger.error(f"Unexpected error processing entry: {e}")

    logger.info(f"Score transformation complete! File saved to {output_path}.")


def split_dataset(
    output_dir_path: Path,
    input_file_path: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> None:
    """Splits a JSONL dataset into train, validation, and test sets.

    Args:
        output_dir_path (Path): Directory to save the split dataset files.
        input_file_path (Path): Path to the input JSONL dataset.
        train_ratio (float): Proportion of data for training. Defaults to 0.8.
        val_ratio (float): Proportion of data for validation. Defaults to 0.1.
        test_ratio (float): Proportion of data for testing. Defaults to 0.1.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Raises:
        ValueError: If the split ratios do not sum to 1.
        FileNotFoundError: If the input file does not exist.

    Example:
        >>> split_dataset(Path("data"), Path("dataset.jsonl"), 0.7, 0.2, 0.1)
        INFO: Split complete! Created files:
        INFO: Train (7000 samples): data/dataset_train.jsonl
        INFO: Validation (2000 samples): data/dataset_val.jsonl
        INFO: Test (1000 samples): data/dataset_test.jsonl
    """
    # Validate ratios
    if not sum([train_ratio, val_ratio, test_ratio]) == 1.0:
        raise ValueError(f"Split ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    if not input_file_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file_path}")

    # Count total number of lines
    with input_file_path.open("r", encoding="utf-8") as f:
        num_samples = sum(1 for _ in f)

    # Calculate dataset sizes
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_size = num_samples - train_size - val_size  # Ensures total consistency

    # Create shuffled indices
    random.seed(seed)
    shuffled_indices = list(range(num_samples))
    random.shuffle(shuffled_indices)

    # Define split boundaries
    train_indices = set(shuffled_indices[:train_size])
    val_indices = set(shuffled_indices[train_size : train_size + val_size])

    # Ensure output directory exists
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    train_file = output_dir_path / f"{input_file_path.stem}_train.jsonl"
    val_file = output_dir_path / f"{input_file_path.stem}_val.jsonl"
    test_file = output_dir_path / f"{input_file_path.stem}_test.jsonl"

    # Open files for writing
    with train_file.open("w", encoding="utf-8") as train_f, val_file.open(
        "w", encoding="utf-8"
    ) as val_f, test_file.open("w", encoding="utf-8") as test_f, input_file_path.open("r", encoding="utf-8") as in_f:
        for idx, line in tqdm(enumerate(in_f), total=num_samples, desc="Splitting dataset"):
            if idx in train_indices:
                train_f.write(line)
            elif idx in val_indices:
                val_f.write(line)
            else:
                test_f.write(line)

    # Log results
    logger.info("Split complete! Created files:")
    logger.info(f"Train ({train_size} samples): {train_file}")
    logger.info(f"Validation ({val_size} samples): {val_file}")
    logger.info(f"Test ({test_size} samples): {test_file}")
