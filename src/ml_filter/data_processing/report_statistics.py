import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from pydantic import BaseModel, ConfigDict

from ml_filter.data_processing.document import Annotation
from ml_filter.data_processing.document_processor import logger


class ReportStats(BaseModel):
    model_config = ConfigDict(extra="allow")
    mae: float
    mse: float
    error_mean_std: float
    acc: float
    predicted_scores_mean_std_var: float


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
                # try to convert single scores to list of scores
                if "id" in line and "score" in line:
                    json_line = json.loads(line)
                    annotation = {"document_id": json_line["id"], "scores": [json_line["score"]]}
                else:
                    raise ValueError(
                        f"Abort statistic computations. Error while parsing annotations from {jsonl_file_path}: {e}"
                    )
            annotations.append(annotation)
    df = pd.DataFrame(annotations)
    df["score"] = df.scores.apply(_get_most_common_score)
    return df


def report_statistics(result_dir_path: Path, gold_annotations_file_paths: List[Path]) -> Dict[str, Any]:
    """Show the comparison between the original and generated score."""

    logger.info(
        "Computing statistics by selecting the most common score when multiple scores per document "
        + "were predicted..."
    )
    # Load all gold annotations across multiple files
    df_gold = pd.concat(list(map(_load_validated_jsonl_as_dataframe, gold_annotations_file_paths)))

    # Load all annotated results across multiple files
    df = pd.concat(list(map(_load_validated_jsonl_as_dataframe, result_dir_path.glob("**/*__annotations_*.jsonl"))))

    # Merge both dataframes on document_id
    stats = df.merge(df_gold, on="document_id", suffixes=("_pred", "_gold"))

    # Check for missing documents after merge
    if len(stats) != len(df_gold):
        raise ValueError("Mismatch in document counts after merging. Some documents are missing.")

    stats["score_mae"] = (stats["score_pred"] - stats["score_gold"]).abs().mean()
    stats["score_mse"] = ((stats["score_pred"] - stats["score_gold"]) ** 2).mean()
    stats["score_std"] = (stats["score_pred"] - stats["score_gold"]).std()
    stats["accuracy"] = (stats["score_pred"] == stats["score_gold"]).mean()

    # Transform the list of document_processing_status to individual columns and perform value_counts on each column
    status_counts = (
        pd.DataFrame(df["document_processing_status"].tolist()).apply(pd.Series.value_counts).fillna(0).astype(int)
    )
    status_counts.index = status_counts.index.astype(str)
    error_counts = pd.DataFrame(df["errors"].tolist()).apply(pd.Series.value_counts)
    error_counts.index = error_counts.index.astype(str)

    statistics_report = {
        **ReportStats(
            mae=stats["score_mae"].mean(),
            mse=stats["score_mse"].mean(),
            error_mean_std=stats["score_std"].mean(),
            acc=stats["accuracy"].mean(),
            predicted_scores_mean_std_var=stats["scores_pred"].apply(lambda x: pd.Series(x).std()).mean(),
        ).model_dump(),
        "confusion_matrix": pd.crosstab(stats["score_gold"], stats["score_pred"]).to_dict(),
        "predicted_score_counts": stats["score_pred"].value_counts().sort_index().to_dict(),
        "gold_score_counts": stats["score_gold"].value_counts().sort_index().to_dict(),
        "document_status_counts": status_counts.to_dict(),
        "error_counts": error_counts.to_dict(),
        "gold_annotations_file_path": str(gold_annotations_file_paths),
        "predicted_annotations_file_path": str(result_dir_path),
    }
    logger.info(json.dumps(statistics_report, indent=4))

    output_dir_path = result_dir_path.parent
    with open(output_dir_path / "statistics_report.json", "w") as f:
        json.dump(statistics_report, f, indent=4)

    return statistics_report
