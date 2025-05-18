import itertools
import logging
from pathlib import Path

from ml_filter.analysis.interrater_reliability import compute_interrater_reliability_metrics, compute_metrics
from ml_filter.analysis.utils import get_common_docs, get_document_scores_df
from ml_filter.utils.logging import get_logger

logger = get_logger(name=__name__, level=logging.INFO)  # Set up logging


def _extract_annotator_name(filename: Path) -> str:
    """
    Extracts the annotator name from the filename.

    Args:
        filename (Path): The path to the file.

    Returns:
        str: The extracted annotator name.
    """
    basename = filename.stem
    # TODO: Split based on '__'
    return basename.split("_")[-1]


def evaluate_predicted_annotations(
    input_directory: Path,
    output_directory: Path,
    path_to_ground_truth_file: Path,
    aggregation: str,
    valid_labels: list[int],
    thresholds: list[float],
) -> None:
    """
    Evaluates prompt-based annotations by comparing annotations to ground truth data.

    Args:
        input_directory (Path): The directory containing the annotation files.
        output_directory (Path): The directory to save the evaluation results.
        gt_data (Path): The path to the ground truth data file.
        aggregation (str): The aggregation method to use for the scores.
        labels (list[float]): The list of possible labels.
        thresholds (list[float]): A list of thresholds for computing agreement metrics.

    Returns:
        None
    """
    # Find all files matching the pattern in the directory and subdirectories
    files = list(input_directory.rglob("annotations_*.jsonl"))

    # Check if there is at least one file
    if len(files) == 0:
        raise ValueError(f"No annotation files found in {input_directory} or its subdirectories.")

    output_directory.mkdir(parents=True, exist_ok=True)

    # Iterate over all pairs of files (tuples)
    for file in files:
        # Extract annotator names
        annotator = _extract_annotator_name(file)
        # NOTE: Here we rely on a specific directory structure
        # TODO: Make this more robust, e.g., by extracting the language from the filename
        lang = file.parent.name

        # Log the tuple of annotator names
        logger.info(f"Compare annotator {annotator} to ground truth.")
        lang_dir = output_directory / lang
        lang_dir.mkdir(parents=True, exist_ok=True)

        compute_interrater_reliability_metrics(
            file_paths=([path_to_ground_truth_file, file]),
            output_dir=lang_dir,
            aggregation_strategy=aggregation,
            valid_labels=valid_labels,
            thresholds=thresholds,
            lang=lang,
        )
        logger.info(f"Metrics successfully written to {lang_dir}")


def evaluate_prediction_correlation(
    input_directory: Path,
    model_filters: list[str],
) -> None:
    # Find all files matching the pattern in the directory and subdirectories
    files = list(input_directory.rglob("annotations_*.jsonl"))

    # Check if there is at least one file
    if len(files) < 2:
        raise ValueError(f"No annotation files found in {input_directory} or its subdirectories.")

    filtered_file_paths = [
        file_path for file_path in files if any(model_filter in str(file_path) for model_filter in model_filters)
    ]

    scores_df = get_document_scores_df(
        input_file_paths=filtered_file_paths,
        aggregation_strategy="majority",
        valid_labels=[0, 1, 2, 3, 4, 5],
    )

    # create all pairs
    model_pairs = list(itertools.combinations(model_filters, 2))
    for model_pair in model_pairs:
        model_1, model_2 = model_pair
        common_docs_df = get_common_docs(scores_df, model_1, model_2)
        valid_docs_df = common_docs_df[
            (common_docs_df["score_0"] != "invalid") & (common_docs_df["score_1"] != "invalid")
        ]
        valid_docs_df = valid_docs_df[valid_docs_df["prompt_lang"] != "en"]

        metrics = compute_metrics(
            num_total_docs=len(common_docs_df),
            valid_docs_df=valid_docs_df,
            thresholds=[1],
        )
        print(float(metrics["metrics"]["Spearman"]))
