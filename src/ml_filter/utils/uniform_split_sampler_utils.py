import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

logger = logging.getLogger(__name__)


def load_sampler_config(config_path: Path) -> Dict[str, Any]:
    if not config_path.exists():
        raise FileNotFoundError(f"Sampler config file not found: {config_path}")

    logger.info("Loading sampler configuration from %s", config_path)
    config_container = OmegaConf.load(config_path)
    config = OmegaConf.to_container(config_container, resolve=True)
    if not isinstance(config, dict):
        raise ValueError(f"Sampler configuration must be a mapping, but received {type(config_container)}")
    return config


def extract_score_value(value: Any) -> float | int | np.floating | np.integer | None:
    """Extract a scalar score from lists/arrays when present."""

    if isinstance(value, (list, tuple, np.ndarray)):
        return value[0] if len(value) > 0 else np.nan
    return value  # type: ignore[return-value]


def split_label_pools(
    df: pd.DataFrame,
    unique_scores: List[float],
    score_column: str,
    validation_fraction: float,
    random_seed: int,
) -> Tuple[Dict[float, pd.DataFrame], Dict[float, pd.DataFrame]]:
    train_pools: Dict[float, pd.DataFrame] = {}
    val_pools: Dict[float, pd.DataFrame] = {}

    grouped = df.groupby(score_column)
    for score in unique_scores:
        label_data = grouped.get_group(score) if score in grouped.groups else df.head(0).copy()
        if label_data.empty:
            base = df.head(0).copy()
            train_pools[score] = base.copy()
            val_pools[score] = base.copy()
            continue

        shuffled = label_data.sample(frac=1, replace=False, random_state=random_seed + int(score)).reset_index(
            drop=True
        )
        n_total = len(shuffled)
        n_train_rows = int(np.floor(n_total * (1 - validation_fraction)))
        n_val_rows = n_total - n_train_rows

        train_pools[score] = shuffled.iloc[:n_train_rows]
        val_pools[score] = shuffled.iloc[n_train_rows : n_train_rows + n_val_rows]

    return train_pools, val_pools


def sample_with_cap(
    pool: pd.DataFrame,
    target: int,
    score: float,
    split_name: str,
    seed_offset: int,
    random_seed: int,
    max_upsample_factor: float,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    if pool.empty or target <= 0:
        return pool.head(0).copy()

    max_allowed = int(len(pool) * max_upsample_factor)
    if max_allowed == 0:
        return pool.head(0).copy()

    effective_target = min(target, max_allowed)

    replace = len(pool) < effective_target
    sample = pool.sample(
        n=effective_target,
        replace=replace,
        random_state=random_seed + seed_offset + int(score),
    )
    if replace:
        factor = effective_target / len(pool)
        if log:
            log.info(
                "  Oversampling %s %.1f: %d → %d (%.1fx)",
                split_name,
                score,
                len(pool),
                effective_target,
                factor,
            )
    return sample


def per_label_targets(scores: List[float], total_target: int) -> Dict[float, int]:
    if not scores or total_target <= 0:
        return {score: 0 for score in scores}

    base = total_target // len(scores)
    remainder = total_target - base * len(scores)

    targets: Dict[float, int] = {}
    for idx, score in enumerate(scores):
        targets[score] = base + (1 if idx < remainder else 0)
    return targets


def save_dataset(df: pd.DataFrame, path: Path, score_column: str, log: logging.Logger | None = None) -> None:
    df_to_write = df.copy()
    if not df_to_write.empty:
        df_to_write[score_column] = df_to_write[score_column].apply(lambda x: [x])
    df_to_write.to_json(path, orient="records", lines=True)
    if log:
        log.info("Wrote %d rows to %s", len(df), path)


def log_distribution(
    df: pd.DataFrame,
    score_column: str,
    label: str,
    target_total: float | None = None,
    log: logging.Logger | None = None,
) -> None:
    log_ref = log or logger
    if df.empty:
        log_ref.info("%s: 0 samples", label)
        return

    counts = df[score_column].value_counts().sort_index()
    total = len(df)
    log_ref.info("%s: %d samples", label, total)
    for score, count in counts.items():
        split_pct = (count / total) * 100 if total > 0 else 0
        if target_total and target_total > 0:
            target_pct = (count / target_total) * 100
            log_ref.info(
                "  Score %.1f: %d (%.2f%% of split, %.2f%% of target)",
                score,
                count,
                split_pct,
                target_pct,
            )
        else:
            log_ref.info("  Score %.1f: %d (%.2f%%)", score, count, split_pct)
