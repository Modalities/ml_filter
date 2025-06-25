"""
Sampling Script - Balanced Distribution

This script processes JSONL files containing scored data and creates a 
distribution based on the Hessen AI's distribution - keeping common 
classes roughly equal while moderately boosting rare classes, then splits the data 
into training and validation sets while preventing ID leakage.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class BalancedDataSampler:
    """Handles balanced distribution sampling - Mimics Hessen AI's distributions."""

    def __init__(
        self,
        input_dir: str,
        output_dir: str,
        total_samples: int = 500_000,
        validation_fraction: float = 0.10,
        score_column: str = "score",
        random_seed: int = 42,
    ):
        """
        Initialize the BalancedDataSampler.

        Args:
            input_dir: Directory containing input JSONL files
            output_dir: Base directory for output files
            total_samples: Total number of samples to generate
            validation_fraction: Fraction of data to use for validation
            score_column: Name of the column containing scores
            random_seed: Random seed for reproducibility
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.total_samples = total_samples
        self.validation_fraction = validation_fraction
        self.score_column = score_column
        self.random_seed = random_seed

        # Set random seed
        np.random.seed(random_seed)

        # Create output directories
        self.train_dir = self.output_dir / "training_set"
        self.val_dir = self.output_dir / "validation_set"

        for directory in [self.train_dir, self.val_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def load_and_validate_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load and validate a JSONL file.

        Args:
            file_path: Path to the JSONL file

        Returns:
            Validated DataFrame with integer scores
        """
        try:
            df = pd.read_json(file_path, lines=True)

            if self.score_column not in df.columns:
                raise ValueError(f"Missing required column: {self.score_column}")

            # Convert scores to numeric and remove invalid entries
            df[self.score_column] = pd.to_numeric(df[self.score_column], errors="coerce")
            df = df.dropna(subset=[self.score_column])

            # Keep only values where int(number) == float(number), as reported by Hessen AI
            df = df[df[self.score_column].apply(lambda x: int(x) == float(x))]

            # Add language identifier
            language = self._extract_language(file_path.name)
            df["language"] = language

            logger.info(f"Loaded {len(df)} valid samples from {file_path.name}")
            return df

        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return pd.DataFrame()

    def _extract_language(self, filename: str) -> str:
        """Extract language code from filename."""
        return filename.split("_sampled", 1)[0]

    def create_balanced_sample_and_split(self, df: pd.DataFrame, target_size: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create a balanced distribution that's a middle ground between natural and uniform.
        - Common classes (0-3): Keep roughly equal (mild oversampling of underrepresented)
        - Rare classes (4-5): Moderate to heavy oversampling, but not full uniformity

        Args:
            df: Input DataFrame
            target_size: Target number of samples per language

        Returns:
            Tuple of (training_df, validation_df) with balanced distributions
        """
        unique_scores = sorted(df[self.score_column].unique())

        # Calculate natural distribution first to understand the data
        score_counts = df[self.score_column].value_counts()
        total_available = len(df)

        logger.info("Original distribution:")
        for score in unique_scores:
            count = score_counts.get(score, 0)
            pct = (count / total_available) * 100
            logger.info(f"  Score {score}: {count} samples ({pct:.2f}%)")

        # Define balanced target strategy
        total_val_target = int(target_size * self.validation_fraction)
        total_train_target = target_size - total_val_target

        # Strategy: Create a middle-ground distribution
        target_proportions = self._calculate_balanced_proportions(unique_scores, score_counts, total_available)

        # Calculate targets based on balanced proportions
        train_targets = {}
        val_targets = {}

        logger.info("Balanced target distribution:")
        for score in unique_scores:
            proportion = target_proportions[score]
            train_targets[score] = int(total_train_target * proportion)
            val_targets[score] = int(total_val_target * proportion)

            available = score_counts.get(score, 0)
            total_target = train_targets[score] + val_targets[score]
            boost_factor = total_target / available if available > 0 else 0

            logger.info(f"  Score {score}: {total_target} target ({proportion*100:.1f}%) - {boost_factor:.1f}x boost")

        # Adjust for rounding to hit exact target
        self._adjust_targets_for_rounding(
            train_targets, val_targets, total_train_target, total_val_target, unique_scores
        )

        train_parts = []
        val_parts = []

        np.random.seed(self.random_seed)

        for score in unique_scores:
            score_data = df[df[self.score_column] == score]
            available_count = len(score_data)

            if available_count == 0:
                continue

            train_target = train_targets[score]
            val_target = val_targets[score]

            logger.info(f"Score {score}: {available_count} available → Train: {train_target}, Val: {val_target}")

            # Get unique IDs for this score
            unique_ids = score_data["id"].unique()
            n_unique = len(unique_ids)

            # Split unique IDs between train and validation
            n_val_ids = max(1, int(n_unique * self.validation_fraction)) if val_target > 0 else 0
            n_train_ids = n_unique - n_val_ids

            # Handle edge cases
            if n_unique == 1 and train_target > 0 and val_target > 0:
                if train_target >= val_target:
                    train_ids = set(unique_ids)
                    val_ids = set()
                    val_target = 0
                else:
                    train_ids = set()
                    val_ids = set(unique_ids)
                    train_target = 0
            else:
                shuffled_ids = unique_ids.copy()
                np.random.shuffle(shuffled_ids)

                train_ids = set(shuffled_ids[:n_train_ids])
                val_ids = set(shuffled_ids[n_train_ids:])

            # Create train set with balanced oversampling
            if train_target > 0 and len(train_ids) > 0:
                train_data = score_data[score_data["id"].isin(train_ids)]
                if len(train_data) < train_target:
                    train_sample = train_data.sample(
                        n=train_target, replace=True, random_state=self.random_seed + int(score)
                    )
                    factor = train_target / len(train_data)
                    logger.info(f"  Oversampling Train {score}: {len(train_data)} → {train_target} ({factor:.1f}x)")
                else:
                    train_sample = train_data.sample(
                        n=train_target, replace=False, random_state=self.random_seed + int(score)
                    )
                train_parts.append(train_sample)

            # Create validation set with balanced oversampling
            if val_target > 0 and len(val_ids) > 0:
                val_data = score_data[score_data["id"].isin(val_ids)]
                if len(val_data) < val_target:
                    val_sample = val_data.sample(
                        n=val_target, replace=True, random_state=self.random_seed + int(score) + 1000
                    )
                    factor = val_target / len(val_data)
                    logger.info(f"  Oversampling Val {score}: {len(val_data)} → {val_target} ({factor:.1f}x)")
                else:
                    val_sample = val_data.sample(
                        n=val_target, replace=False, random_state=self.random_seed + int(score) + 1000
                    )
                val_parts.append(val_sample)

        # Combine and shuffle
        train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame()
        val_df = pd.concat(val_parts, ignore_index=True) if val_parts else pd.DataFrame()

        if not train_df.empty:
            train_df = train_df.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        if not val_df.empty:
            val_df = val_df.sample(frac=1, random_state=self.random_seed + 2000).reset_index(drop=True)

        # Verification
        if not train_df.empty and not val_df.empty:
            train_ids = set(train_df["id"])
            val_ids = set(val_df["id"])
            overlap = train_ids.intersection(val_ids)

            if overlap:
                logger.error(f"CRITICAL: {len(overlap)} IDs appear in both train and validation!")
                raise ValueError("IDs found in both train and validation sets!")

        logger.info(f"✅ Balanced distribution achieved: Train: {len(train_df)} samples, Val: {len(val_df)} samples")
        return train_df, val_df

    def _calculate_balanced_proportions(self, unique_scores, score_counts, total_available):
        """Calculate target proportions to match the paper's distribution."""

        # Target proportions mimics Hessen AI's distribution:
        # Class 0: 115610, Class 1: 115716, Class 2: 116128, Class 3: 112935, Class 4: 39017, Class 5: 654
        # Total: ~500k, so proportions are:
        target_proportions = {
            0: 0.2312,  # ~23.12%
            1: 0.2314,  # ~23.14%
            2: 0.2323,  # ~23.22%
            3: 0.2259,  # ~22.58%
            4: 0.0780,  # ~7.80%
            5: 0.0013,  # ~0.13%
        }

        # Use target proportions if score exists, otherwise fall back to natural
        balanced_proportions = {}

        for score in unique_scores:
            score_int = int(score)
            if score_int in target_proportions:
                balanced_proportions[score] = target_proportions[score_int]
            else:
                # Fallback for unexpected scores
                natural_proportion = score_counts.get(score, 0) / total_available
                balanced_proportions[score] = natural_proportion

        # Normalize to ensure they sum to 1.0 (in case we have missing/extra scores)
        total_prop = sum(balanced_proportions.values())
        if total_prop > 0:
            for score in balanced_proportions:
                balanced_proportions[score] /= total_prop

        return balanced_proportions

    def _adjust_targets_for_rounding(
        self, train_targets, val_targets, total_train_target, total_val_target, unique_scores
    ):
        """Adjust targets to hit exact totals after rounding."""
        # Calculate current totals
        current_train_total = sum(train_targets.values())
        current_val_total = sum(val_targets.values())

        # Distribute remainders to most common classes first
        train_remainder = total_train_target - current_train_total
        val_remainder = total_val_target - current_val_total

        # Add remainders to first few classes
        for i, score in enumerate(unique_scores):
            if i < train_remainder:
                train_targets[score] += 1
            if i < val_remainder:
                val_targets[score] += 1

    def _prepare_for_output(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare DataFrame for JSON output by wrapping scores in lists."""
        output_df = df.copy()
        output_df[self.score_column] = output_df[self.score_column].apply(lambda x: [x])
        return output_df

    def save_datasets(self, train_df: pd.DataFrame, val_df: pd.DataFrame, base_filename: str) -> Dict[str, int]:
        """
        Save training and validation datasets.

        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            base_filename: Base name for output files

        Returns:
            Dictionary with dataset sizes
        """
        base_name = base_filename.replace(".jsonl", "")

        train_output = self._prepare_for_output(train_df)
        val_output = self._prepare_for_output(val_df)

        train_path = self.train_dir / f"{base_name}_train.jsonl"
        val_path = self.val_dir / f"{base_name}_val.jsonl"

        train_output.to_json(train_path, orient="records", lines=True)
        val_output.to_json(val_path, orient="records", lines=True)

        return {"train": len(train_df), "validation": len(val_df)}

    def _log_dataset_stats(self, df: pd.DataFrame, dataset_name: str, language: str):
        """Log statistics for a dataset."""
        if df.empty:
            logger.info(f"{dataset_name} ({language}): 0 samples")
            return

        score_counts = df[self.score_column].value_counts().sort_index()
        total_samples = len(df)

        logger.info(f"{dataset_name} ({language}): {total_samples} samples")
        for score, count in score_counts.items():
            percentage = (count / total_samples) * 100
            logger.info(f"  Score {score}: {count} samples ({percentage:.2f}%)")

    def process_all_files(self):
        """Process all JSONL files in the input directory."""
        jsonl_files = list(self.input_dir.glob("*.jsonl"))

        if not jsonl_files:
            logger.error("No JSONL files found in input directory")
            return

        # Load all valid datasets
        datasets = []
        for file_path in jsonl_files:
            df = self.load_and_validate_data(file_path)
            if not df.empty:
                datasets.append((file_path.name, df))

        if not datasets:
            logger.error("No valid datasets found")
            return

        # Calculate samples per language
        n_languages = len(datasets)
        samples_per_language = self.total_samples // n_languages

        logger.info(f"Processing {n_languages} languages with {samples_per_language} samples each")
        logger.info("Using BALANCED distribution - middle ground between natural and uniform")

        # Process each dataset
        for filename, df in datasets:
            language = df["language"].iloc[0]
            logger.info(f"\nProcessing {filename} (Language: {language})")

            # Create balanced sample and split in one step (prevents ID leakage)
            # This uses middle-ground approach: balance common classes, boost rare classes moderately
            train_df, val_df = self.create_balanced_sample_and_split(df, samples_per_language)

            # Save datasets
            self.save_datasets(train_df, val_df, filename)

            # Log statistics
            self._log_dataset_stats(train_df, "Training", language)
            self._log_dataset_stats(val_df, "Validation", language)

        logger.info(f"\nProcessing complete! Output saved to: {self.output_dir}")


def main():
    """Main execution function."""
    # Configuration
    config = {
        "input_dir": "./fineweb2/Mistral-Small-3.1-24B-Instruct-2503_aggregated/",
        "output_dir": "./processed_data_natural/",
        "total_samples": 500_000,
        "validation_fraction": 0.10,
        "score_column": "score",
        "random_seed": 42,
    }

    # Create and run sampler
    sampler = BalancedDataSampler(**config)
    sampler.process_all_files()


if __name__ == "__main__":
    main()
