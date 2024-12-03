import json
from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from omegaconf import OmegaConf

from ml_filter.data_processing.document_processor import report_statistics


def compare_experiments(config_file_path: Path):
    cfg = OmegaConf.load(config_file_path)
    paths = [Path(path) for path in cfg.experiment_dir_paths]
    path_exists = [path.exists() for path in paths]

    if not all(path_exists):
        raise AssertionError(f"Paths do not exist: {zip(paths, path_exists)}")

    results = []
    for path in paths:
        config = OmegaConf.load(path / "lorem_ipsum.yaml")
        stats = report_statistics(results_file_path=path / "processed_documents.jsonl")
        stats["model_name"] = config.settings.model_name
        stats["add_generation_prompt"] = config.tokenizer.add_generation_prompt
        stats["experiment_path"] = str(path)

        with open(path / "statistics_report.json", "w") as f:
            json.dump(stats, f, indent=4)

        {key: stats[key] for key in stats.keys() if isinstance(stats[key], str) or not isinstance(stats[key], Iterable)}
        results.append(stats)

    df = pd.DataFrame(results)
    df = df.sort_values(by=["mae"]).reset_index(drop=True)

    with open(paths[0].parent / "comparison_report.csv", "w") as f:
        df.to_csv(f, index=False)

    print("Comparison Report:")
    print(df)

    print("Best Model Stats:")
    best_model_stats = df.iloc[0]
    print(best_model_stats)
    print("Confusion Matrix:")
    print(pd.DataFrame(best_model_stats["confusion_matrix"]))
