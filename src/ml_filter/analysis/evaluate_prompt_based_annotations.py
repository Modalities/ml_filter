
import logging
from pathlib import Path
from typing import List
from ml_filter.analysis.interrater_reliability import compute_interrater_reliability_metrics
from ml_filter.utils.logging import get_logger

logger = get_logger(name=__name__, level=logging.INFO) # Set up logging


def extract_model_name(filename: Path) -> str:
    # Function to extract the model name from the filename
    basename = filename.stem
    return basename.split("_")[-1]


def evaluate_prompt_based_annotations(
    input_directory: Path,
    output_directory: Path,
    gt_data: Path,
    aggregation: str,
    labels: List[float]
) -> None:
    # Find all files matching the pattern in the directory and subdirectories
    files = list(input_directory.rglob("annotations_*.jsonl"))

    # Check if there are at least two files
    if len(files) == 0:
        logger.info(f"No annotation files found in {input_directory} or its subdirectories. Exiting.")
        exit(1)

    output_directory.mkdir(parents=True, exist_ok=True)

    # Iterate over all pairs of files (tuples)
    for file in files:
        # Extract model names
        model = extract_model_name(file)
        lang = file.parent.name
        
        # Log the tuple of model names
        logger.info(f"Compare model {model} to ground truth")
        lang_dir = output_directory / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        compute_interrater_reliability_metrics(
            path_to_files=([gt_data, file]),
            output_dir=lang_dir,
            aggregation=aggregation,
            truth_file_idx=0,
            model_name=model,
            labels=labels,
        )
        logger.info(f"Metrics successfully written to {lang_dir}")
