from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel

from ml_filter.data_processing.document_processor import ReportStats, report_statistics


class StatisticConfig(BaseModel):
    sort_by_metric: str
    ascending: bool


class CompareConfig(BaseModel):
    experiment_dir_paths: List[str]
    config_file_name: str
    output_format: List[StatisticConfig]
    gold_annotations_file_path: str


def compare_experiments(config_file_path: Path):
    cfg = OmegaConf.load(config_file_path)
    config = CompareConfig(**OmegaConf.to_container(cfg, resolve=True))
    paths = [Path(path) for path in config.experiment_dir_paths]

    path_exists = [path.exists() for path in paths]
    if not all(path_exists):
        raise ValueError(f"Paths do not exist: {zip(paths, path_exists)}")
    config_filename = config.config_file_name

    allowed_metrics = ReportStats.model_fields.keys()
    for stat in config.output_format:
        if stat.sort_by_metric not in allowed_metrics:
            raise ValueError(f"Allowed metrics to sort by: {allowed_metrics}")

    results = []
    for path in paths:
        exp_config = OmegaConf.load(path / config_filename)
        stats = report_statistics(
            results_file_path=path / "annotations" / "processed_documents.jsonl",
            gold_annotations_file_path=Path(config.gold_annotations_file_path),
        )
        stats["model_name"] = exp_config.settings.model_name
        stats["add_generation_prompt"] = exp_config.tokenizer.add_generation_prompt
        stats["experiment_path"] = str(path)
        results.append(stats)

    df = pd.DataFrame(results)

    # Sort results
    by_values = [stat.sort_by_metric for stat in config.output_format]
    are_ascending = [stat.ascending for stat in config.output_format]
    df = df.sort_values(by=by_values, ascending=are_ascending).reset_index(drop=True)

    with open(paths[0].parent / "comparison_report.csv", "w") as f:
        df.to_csv(f, index=False)

    print("Comparison Report:")
    print(df)

    print("Best Model Stats:")
    best_model_stats = df.iloc[0]
    print(best_model_stats)
    print("Confusion Matrix:")
    print(pd.DataFrame(best_model_stats["confusion_matrix"]))
