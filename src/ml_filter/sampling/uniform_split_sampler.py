"""Uniform split sampler: split by label first, then oversample within each split."""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from ml_filter.utils.uniform_split_sampler_utils import (
    log_distribution,
    normalize_score_value,
    per_label_targets,
    sample_with_cap,
    save_dataset,
    split_label_pools,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
__all__ = ["UniformSplitSampler"]


class UniformSplitSampler:
    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        validation_fraction: float = 0.10,
        score_column: str = "score",
        random_seed: int = 42,
        max_oversampling_ratio: float = 10.0,
        per_label_target: int | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.validation_fraction = validation_fraction
        self.score_column = score_column
        self.random_seed = random_seed
        self.max_oversampling_ratio = max_oversampling_ratio
        self.per_label_target = per_label_target

        self.train_dir = self.output_dir / "training_set"
        self.val_dir = self.output_dir / "validation_set"
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)

        np.random.seed(self.random_seed)

    def process_all_files(self):
        jsonl_files = sorted(self.input_dir.glob("*.jsonl"))
        if not jsonl_files:
            logger.error("No JSONL files found in %s", self.input_dir)
            return

        datasets: List[Tuple[str, pd.DataFrame]] = []
        for path in jsonl_files:
            df = self._load_file(path)
            if not df.empty:
                datasets.append((path.name, df))

        if not datasets:
            logger.error("No valid datasets to process.")
            return

        for filename, df in datasets:
            language = df.get("language", pd.Series(["unknown"])).iloc[0]
            logger.info("\nProcessing %s (%s) with %d available rows", filename, language, len(df))

            target_size = len(df)
            train_df, val_df, train_target_total, val_target_total = self._build_splits(df, target_size)

            save_dataset(
                train_df,
                self.train_dir / f"{filename.replace('.jsonl', '')}_train.jsonl",
                score_column=self.score_column,
                log=logger,
            )
            save_dataset(
                val_df,
                self.val_dir / f"{filename.replace('.jsonl', '')}_val.jsonl",
                score_column=self.score_column,
                log=logger,
            )

            log_distribution(train_df, self.score_column, f"Training ({language})", train_target_total, logger)
            log_distribution(val_df, self.score_column, f"Validation ({language})", val_target_total, logger)

        logger.info("\nAll files processed. Output written to %s", self.output_dir)

    def _load_file(self, file_path: Path) -> pd.DataFrame:
        try:
            df = pd.read_json(file_path, lines=True)
        except ValueError as exc:
            logger.error("Failed to read %s: %s", file_path, exc)
            return pd.DataFrame()

        if self.score_column not in df.columns:
            logger.error("File %s missing required column '%s'", file_path, self.score_column)
            return pd.DataFrame()

        df[self.score_column] = df[self.score_column].apply(normalize_score_value)
        df[self.score_column] = pd.to_numeric(df[self.score_column], errors="coerce")
        df = df.dropna(subset=[self.score_column])

        df = df[df[self.score_column].apply(lambda x: int(x) == float(x))]
        df["language"] = file_path.name.split("_sampled", 1)[0]

        logger.info("Loaded %d valid rows from %s", len(df), file_path.name)
        return df

    def _build_splits(self, df: pd.DataFrame, target_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        unique_scores = sorted(df[self.score_column].unique())
        if not unique_scores:
            return df.head(0).copy(), df.head(0).copy()

        per_label_total_target = (
            float(self.per_label_target) if self.per_label_target is not None else (target_size / len(unique_scores))
        )

        train_target_total = int(per_label_total_target * (1 - self.validation_fraction) * len(unique_scores))
        val_target_total = int(per_label_total_target * self.validation_fraction * len(unique_scores))

        train_targets = per_label_targets(unique_scores, train_target_total)
        val_targets = per_label_targets(unique_scores, val_target_total)

        train_pools, val_pools = split_label_pools(
            df,
            unique_scores,
            score_column=self.score_column,
            validation_fraction=self.validation_fraction,
            random_seed=self.random_seed,
        )

        train_samples = []
        val_samples = []

        for score in unique_scores:
            train_pool = train_pools.get(score, df.head(0).copy())
            val_pool = val_pools.get(score, df.head(0).copy())

            logger.info(
                "Score %.1f → train pool %d rows, val pool %d rows (targets: train %d, val %d)",
                score,
                len(train_pool),
                len(val_pool),
                train_targets.get(score, 0),
                val_targets.get(score, 0),
            )

            train_sample = sample_with_cap(
                train_pool,
                train_targets.get(score, 0),
                score,
                "train",
                seed_offset=0,
                random_seed=self.random_seed,
                max_oversampling_ratio=self.max_oversampling_ratio,
                log=logger,
            )
            val_sample = sample_with_cap(
                val_pool,
                val_targets.get(score, 0),
                score,
                "validation",
                seed_offset=10_000,
                random_seed=self.random_seed,
                max_oversampling_ratio=self.max_oversampling_ratio,
                log=logger,
            )

            if not train_sample.empty:
                train_samples.append(train_sample)
            if not val_sample.empty:
                val_samples.append(val_sample)

        train_df = pd.concat(train_samples, ignore_index=True) if train_samples else df.head(0).copy()
        val_df = pd.concat(val_samples, ignore_index=True) if val_samples else df.head(0).copy()

        if not train_df.empty:
            train_df = train_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        if not val_df.empty:
            val_df = val_df.sample(frac=1, random_state=self.random_seed + 1).reset_index(drop=True)

        return train_df, val_df, train_target_total, val_target_total
