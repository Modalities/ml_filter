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

    markdown_report = _get_markdown_report(df)
    with open(config.experiment_dir_paths[0].parent / f"comparison_report__{hash_value}.md", "w") as f:
        f.write(markdown_report)

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


def _get_markdown_report(df: pd.DataFrame) -> str:
    # Determine ranking column
    ranking_columns = ["acc", "mae", "mse"]
    for col in ranking_columns:
        if col in df.columns:
            ranking_column = col
            break
    else:
        ranking_column = None

    if ranking_column is None:
        # No suitable ranking column found
        print("No suitable ranking column found in dataframe.")
    else:
        # Sort and rank the dataframe based on the ranking column
        if ranking_column == "acc":
            # For accuracy, higher is better
            df = df.sort_values(by=ranking_column, ascending=False).reset_index(drop=True)
        else:
            # For mae and mse, lower is better
            df = df.sort_values(by=ranking_column, ascending=True).reset_index(drop=True)

    # Add rank column
    df["Rank"] = df.index + 1

    # Create markdown content
    markdown_lines = []

    markdown_lines.append("# Experiment Report\n")

    # Scalar columns
    scalar_columns = [
        "Rank",
        "mae",
        "mse",
        "std",
        "acc",
        "gold_annotations_file_path",
        "predicted_annotations_file_path",
        "model_name",
        "add_generation_prompt",
        "experiment_path",
        "results_written",
        "elapsed_time_s",
        "results_per_second",
        "mean_out_tokens_per_second",
        "model_name",
        "queue_size",
        "num_processes",
        "max_new_tokens",
    ]

    # Ensure scalar columns exist
    scalar_columns = [col for col in scalar_columns if col in df.columns]

    # Create scalar metrics table
    markdown_lines.append("## Scalar Metrics Across Experiments\n")

    scalar_df = df[scalar_columns].T
    # bold all labels in the index and use Rank as new columns
    df.index = [f"**{idx}**" for idx in df.index]
    scalar_df.columns = df["Rank"]
    scalar_df.columns.name = "Rank"

    scalar_table = scalar_df.to_markdown(index=True)
    markdown_lines.append(scalar_table)
    markdown_lines.append("\n")

    # Sub-df columns
    sub_df_columns = [
        "predicted_score_counts",
        "gold_score_counts",
        "success_rate",
    ]

    # Ensure sub-df columns exist
    sub_df_columns = [col for col in sub_df_columns if col in df.columns]

    # Initialize dictionary to collect combined sub-df data
    sub_df_combined = {sub_col: [] for sub_col in sub_df_columns}

    # For each experiment, collect sub-dfs
    for idx, row in df.iterrows():
        rank = row["Rank"]
        model_name = row.get("model_name", "N/A")

        for sub_col in sub_df_columns:
            sub_data = row[sub_col]

            if isinstance(sub_data, pd.DataFrame):
                # Add Rank and model_name columns to sub_data
                sub_data = sub_data.copy()
                sub_data["Rank"] = rank
                sub_data["model_name"] = model_name
                # Reorder columns to have Rank and model_name first
                cols = ["Rank", "model_name"] + [col for col in sub_data.columns if col not in ["Rank", "model_name"]]
                sub_data = sub_data[cols]
                sub_df_combined[sub_col].append(sub_data)

            elif isinstance(sub_data, pd.Series):
                # Convert Series to DataFrame
                sub_df = sub_data.to_frame().T  # Transpose to make it a single row DataFrame
                sub_df["Rank"] = rank
                sub_df["model_name"] = model_name
                # Reorder columns
                cols = ["Rank", "model_name"] + [col for col in sub_df.columns if col not in ["Rank", "model_name"]]
                sub_df = sub_df[cols]
                sub_df_combined[sub_col].append(sub_df)

            elif isinstance(sub_data, dict):
                # Convert dict to DataFrame
                # Handle both scalar and nested dictionaries
                if all(isinstance(v, (int, float, str, bool)) for v in sub_data.values()):
                    sub_df = pd.DataFrame([sub_data])
                else:
                    sub_df = pd.json_normalize(sub_data)
                sub_df["Rank"] = rank
                sub_df["model_name"] = model_name
                # Reorder columns
                cols = ["Rank", "model_name"] + [col for col in sub_df.columns if col not in ["Rank", "model_name"]]
                sub_df = sub_df[cols]
                sub_df_combined[sub_col].append(sub_df)

            else:
                # If sub_data is not a DataFrame, Series, or dict, skip or handle accordingly
                continue

    # After collecting all sub-dfs, combine and add to markdown
    for sub_col in sub_df_columns:
        combined_dfs = sub_df_combined[sub_col]
        if combined_dfs:
            combined_df = pd.concat(combined_dfs, ignore_index=True)
            markdown_lines.append(f"\n## {sub_col} Across Experiments\n")
            sub_df_md = combined_df.to_markdown(index=False)
            markdown_lines.append(sub_df_md)
        else:
            markdown_lines.append(f"\n## {sub_col} Across Experiments\n")
            markdown_lines.append("No data available.")

    not_combine_columns = [col for col in ["confusion_matrix", "error_counts"] if col in df.columns]
    for idx, row in df.iterrows():
        rank = row["Rank"]
        model_name = row.get("model_name", "N/A")
        experiment_path = row.get("experiment_path", "N/A")
        markdown_lines.append(f"\n## {model_name} ranked {rank} in Exp {experiment_path}\n")
        for sub_col in not_combine_columns:
            sub_data = row[sub_col]
            sub_df = pd.DataFrame(sub_data)
            markdown_lines.append(f"\n### {sub_col}\n")
            sub_df_md = sub_df.to_markdown(index=False)
            markdown_lines.append(sub_df_md + "\n")

    # Combine markdown lines
    markdown_content = "\n".join(markdown_lines)

    return markdown_content
