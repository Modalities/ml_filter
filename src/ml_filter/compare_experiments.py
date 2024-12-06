import hashlib
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel, DirectoryPath, FilePath

from ml_filter.data_processing.report_statistics import ReportStats, report_statistics


class StatisticConfig(BaseModel):
    sort_by_metric: str
    ascending: bool


class CompareConfig(BaseModel):
    experiment_dir_paths: List[DirectoryPath]
    experiment_config_file_name: str
    output_format: List[StatisticConfig]
    gold_annotations_file_paths: List[FilePath]


def compare_experiments(config_file_path: Path) -> pd.DataFrame:
    cfg = OmegaConf.load(config_file_path)
    config = CompareConfig(**OmegaConf.to_container(cfg, resolve=True))

    config_filename = config.experiment_config_file_name

    allowed_metrics = ReportStats.model_fields.keys()
    for stat in config.output_format:
        if stat.sort_by_metric not in allowed_metrics:
            raise ValueError(f"Unkown metric to sorty by. Allowed metrics to sort by: {allowed_metrics}")

    results = []
    for result_dir_path in config.experiment_dir_paths:
        exp_config = OmegaConf.load(result_dir_path / config_filename)
        stats = report_statistics(
            result_dir_path=result_dir_path,
            gold_annotations_file_paths=config.gold_annotations_file_paths,
        )
        stats["model_name"] = exp_config.settings.model_name
        stats["add_generation_prompt"] = exp_config.tokenizer.add_generation_prompt
        stats["experiment_path"] = str(result_dir_path)
        results.append(stats)

    df = pd.DataFrame(results)

    # Sort results
    by_values = [stat.sort_by_metric for stat in config.output_format]
    are_ascending = [stat.ascending for stat in config.output_format]
    df = df.sort_values(by=by_values, ascending=are_ascending).reset_index(drop=True)

    with open(config_file_path, "rb") as f:
        hash_value = hashlib.sha256(f.read()).hexdigest()[:8]

    with open(config.experiment_dir_paths[0].parent / f"comparison_report__{hash_value}.csv", "w") as f:
        df.to_csv(f, index=False)

    print("Comparison Report:")
    print(df)

    print("Best Model Stats:")
    best_model_stats = df.iloc[0]
    print(best_model_stats)
    print("Confusion Matrix:")
    print(pd.DataFrame(best_model_stats["confusion_matrix"]))
    return df
