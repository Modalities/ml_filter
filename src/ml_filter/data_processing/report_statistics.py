import json
import logging
from collections import Counter
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel

from ml_filter.data_processing.document import Annotation, DocumentProcessingStatus

# Set up logging
logging.basicConfig(level=logging.INFO)  # Set the logging level as needed
logger = logging.getLogger(__name__)  # Create a logger instance


class ThroughputStatistics(BaseModel):
    """A class representing the throughput statistics."""

    mean_out_tokens_per_second: float
    num_documents_written: int
    elapsed_time_s: float
    documents_per_second: float
    model_name: str
    queue_size: int
    num_processes: int
    max_new_tokens: int


class ThroughputStatisticReport(ThroughputStatistics):
    """A class representing the throughput statistics report."""

    mean_regex_match_rate: float
    experiment_path: str
    num_gpus: int


def _get_most_common_score(scores: List[float | None]) -> float | None:
    """Get the most common score from a list of scores."""
    most_common_score = Counter([float(score) for score in scores if score is not None]).most_common(1)
    if most_common_score:
        return most_common_score[0][0]
    else:
        return None


def _load_validated_jsonl_as_dataframe(jsonl_file_path: Path) -> pd.DataFrame:
    """Load a jsonl file as a pandas DataFrame.
    If the JSON entries are not valid, try to convert single scores to the minimum required format.

    Args:
        jsonl_file_path (Path): The path to the jsonl file.

    Raises:
        ValueError: If the JSON entries are not valid.

    Returns:
        pd.DataFrame: The DataFrame containing the annotations.
    """
    annotations = []
    with open(jsonl_file_path, "r") as f:
        for line in f:
            try:
                annotation = Annotation.model_validate_json(line).model_dump()
            except Exception as e:
                # minimal set for human annotatations
                entry = json.loads(line)
                if "document_id" in entry.keys() and "scores" in entry.keys():
                    annotation = {"document_id": entry["document_id"], "scores": entry["scores"]}
                else:
                    raise ValueError(
                        f"Abort statistic computations. Error while parsing annotations from {jsonl_file_path}: {e}"
                    )
            annotations.append(annotation)
    df = pd.DataFrame(annotations)
    df["score"] = df.scores.apply(_get_most_common_score)
    return df


def report_throughput_statistics(result_dir_path: Path, exp_config_filename: str) -> ThroughputStatisticReport:
    """Show the comparison between the original and generated score."""

    exp_config = OmegaConf.load(result_dir_path / exp_config_filename)

    logger.info(
        "Computing statistics by selecting the most common score when multiple scores per document "
        + "were predicted..."
    )

    # Load all annotated results across multiple files
    df = pd.concat(list(map(_load_validated_jsonl_as_dataframe, result_dir_path.glob("**/*__annotations_*.jsonl"))))

    # Transform the list of document_processing_status to individual columns and perform value_counts on each column
    status_counts = (
        pd.DataFrame(df["document_processing_status"].tolist()).apply(pd.Series.value_counts).fillna(0).astype(int)
    )
    if len(status_counts.columns) > 0:
        status_counts.index = [DocumentProcessingStatus(x).value for x in status_counts.index]
        if "success" in status_counts.index:
            mean_regex_match_rate = (status_counts.loc["success"] / status_counts.sum()).mean()
    else:
        mean_regex_match_rate = 0.0

    # load throughput information
    with open(result_dir_path / "throughput.json", "r") as f:
        throughput = json.load(f)

    statistics_report = ThroughputStatisticReport(
        mean_regex_match_rate=mean_regex_match_rate,
        experiment_path=str(result_dir_path),
        num_gpus=exp_config.settings.num_gpus,
        **throughput,
    )
    out_file_path = result_dir_path / "statistics_report.json"
    with open(out_file_path, "w") as f:
        json.dump(statistics_report.model_dump(), f, indent=4)
    logger.info(f"Statistics report saved to {out_file_path}")

    return statistics_report
