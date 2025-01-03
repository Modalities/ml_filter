import hashlib
from pathlib import Path
from typing import List

import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel, DirectoryPath

from ml_filter.data_processing.report_statistics import report_throughput_statistics


class StatisticConfig(BaseModel):
    sort_by_metric: str
    ascending: bool


class CompareConfig(BaseModel):
    experiment_dir_paths: List[DirectoryPath]
    experiment_config_file_name: str
    output_format: List[StatisticConfig]


def compare_experiments(config_file_path: Path) -> pd.DataFrame:
    cfg = OmegaConf.load(config_file_path)
    config = CompareConfig(**OmegaConf.to_container(cfg, resolve=True))

    results = []
    for result_dir_path in config.experiment_dir_paths:
        stats = report_throughput_statistics(
            result_dir_path=result_dir_path,
            exp_config_filename=config.experiment_config_file_name,
        ).model_dump()
        results.append(stats)
    df = pd.DataFrame(results)

    scalar_columns = [col for col in df.columns if pd.api.types.is_scalar(df[col].iloc[0])]
    for stat in config.output_format:
        if stat.sort_by_metric not in scalar_columns:
            raise ValueError(f"Unkown metric to sorty by. Allowed metrics to sort by: {scalar_columns}")

    # Sort results
    by_values = [stat.sort_by_metric for stat in config.output_format]
    are_ascending = [stat.ascending for stat in config.output_format]
    df = df.sort_values(by=by_values, ascending=are_ascending).reset_index(drop=True)

    with open(config_file_path, "rb") as f:
        hash_value = hashlib.sha256(f.read()).hexdigest()[:8]

    markdown_report = _get_markdown_report(df)
    out_file_path = config.experiment_dir_paths[0].parent / f"comparison_report__{hash_value}.md"
    with open(out_file_path, "w") as f:
        f.write(markdown_report)
    print(f"Markdown report saved to: {out_file_path}")

    out_file_path = config.experiment_dir_paths[0].parent / f"comparison_report__{hash_value}.csv"
    with open(out_file_path, "w") as f:
        df.to_csv(f, index=False)
    return df


def _get_markdown_report(df: pd.DataFrame) -> str:
    # Add rank column
    df["rank"] = df.index + 1

    # Create markdown content
    markdown_lines = []

    markdown_lines.append("# Throughput Report\n")
    scalar_columns = [col for col in df.columns if pd.api.types.is_scalar(df[col].iloc[0])]

    # Create scalar metrics table
    markdown_lines.append("## Throughput Across Experiments\n")

    # bold all labels in the index and use rank as new columns
    scalar_df = df[scalar_columns].T
    scalar_df.columns = df["rank"]
    scalar_df.columns.name = "rank"

    scalar_table = scalar_df.to_markdown(index=True)
    markdown_lines.append(scalar_table)
    markdown_lines.append("\n")

    # Combine markdown lines
    markdown_content = "\n".join(markdown_lines)

    return markdown_content
