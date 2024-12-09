from itertools import combinations
from typing import Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from ml_filter.analysis.utils import get_document_scores


def plot_scores(path_to_files: Tuple[Path], output_dir: Path, aggregation: Union[None, str]) -> None:
    """
    Plots score distributions for each prompt based on the input JSONL files.

    Args:
        path_to_files (Tuple[Path]): A list of paths to JSONL files containing document scores.
        output_dir (Path): The directory to save the generated plots.
        aggregation (Union[None, str]): Aggregation method for scores ("min", "max", "mean", "majority", or None).

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    document_scores = get_document_scores(
        path_to_files=path_to_files,
        aggregation=aggregation
    )
    
    # Iterate over different prompts
    for prompt in document_scores:
        df = _prepare_df(document_scores[prompt], aggregation=aggregation)

        # Plotting the distributions
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(10, 6))
        for model_name in df.columns:
            scores = df[model_name]
            # Create a histogram for each file's scores
            plt.hist(scores, bins=30, alpha=0.5, label=model_name, edgecolor='black')

        # Add labels, title, and annotations
        plt.xlabel(f'{prompt} Score')
        plt.ylabel('Frequency')
        plt.title(f'{prompt} Score Distributions - Aggregation Method {aggregation}')

        # Place annotation below the legend in the upper right corner
        plt.legend(loc='upper right')
        plt.annotate(
            f'Number of Documents: {len(df)}',
            xy=(0.95, 0.85), xycoords='axes fraction',  # Adjust y-coordinate to be slightly below the legend
            fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
        )

        # Save and close the plot
        plt.savefig(output_dir / (prompt + f'_score_distributions_{aggregation}.png'))
        plt.close()


def plot_differences_in_scores(path_to_files: Tuple[Path], output_dir: Path, aggregation: Union[None, str]) -> None:
    """
    Plots histograms and boxplots of score differences between different versions of models.

    Args:
        path_to_files (Tuple[Path]): A list of paths to JSONL files containing document scores.
        output_dir (Path): The directory to save the generated plots.
        aggregation (Union[None, str]): Aggregation method for scores ("min", "max", "mean", "majority", or None).

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    document_scores = get_document_scores(
        path_to_files=path_to_files,
        aggregation=aggregation
    )
    
    # Iterate over different prompts
    for prompt in document_scores:
        df = _prepare_df(document_scores[prompt], aggregation=aggregation)
        
        # Initialize a list to store the differences for each consecutive version pair
        score_differences = {}
            
        # Compute differences for each document for all pairs of versions 
        for (version1, version2) in combinations(df.columns, 2):
            # Compute the difference between the two versions for all documents that have scores in both versions
            differences = df[version2] - df[version1]
            score_differences[(version1, version2)] = differences

        sns.set_theme(style='whitegrid')
        # Plotting the differences as histograms where x-axis represents the difference and y-axis represents the frequency
        plt.figure(figsize=(12, 8))
        
        # Loop through the computed differences and plot histograms for each consecutive version pair
        for i, (version1, version2) in enumerate(score_differences):
            differences = score_differences[(version1, version2)]
            plt.subplot(len(score_differences), 1, i + 1)
            plt.hist(differences, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel('Score Difference')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {prompt} Score Differences ({version2} - {version1}) - Aggregation Method {aggregation}')
            plt.annotate(f'Number of Documents: {len(differences)}', xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

        plt.tight_layout()
        plt.savefig(output_dir / (prompt + f'_score_distributions_difference_histogram_{aggregation}.png'))
        plt.close()

        # Plot boxplot of the score differences
        sns.reset_defaults()
        plt.figure(figsize=(12, 8))
        labels = [f'{version2} - {version1}' for (version1, version2) in score_differences]
        score_differences_values = list(score_differences.values())
        plt.boxplot(score_differences_values, labels=labels, showmeans=True, meanprops=dict(marker='x', markerfacecolor='red', markeredgecolor='red'))
        
        plt.figure(figsize=(10, 6))
        plt.boxplot(score_differences_values, labels=labels, showmeans=True)
        mean_value = np.mean(score_differences_values)
        median_value = np.median(score_differences_values)
        q1 = np.percentile(score_differences_values, 25)
        q3 = np.percentile(score_differences_values, 75)

        plt.annotate(f'Median: {median_value:.2f}', xy=(i + 1, median_value), xytext=(i + 1.1, median_value),
                        fontsize=9, ha='left', va='center', arrowprops=dict(facecolor='black', arrowstyle='->'))
        plt.annotate(f'Mean: {mean_value:.2f}', xy=(i + 1, mean_value), xytext=(i + 0.9, mean_value),
                        fontsize=9, ha='right', va='center', arrowprops=dict(facecolor='red', arrowstyle='->'))
        # Get the current y-axis limits
        y_min, y_max = plt.ylim()
        q1_text_y = max(y_min, min(q1 - 0.5, y_max))
        plt.annotate(f'Q1: {q1:.2f}', xy=(i + 1, q1), xytext=(i + 1.1, q1_text_y),
                        fontsize=9, ha='left', va='center', arrowprops=dict(facecolor='blue', arrowstyle='->'))
        q3_text_y = max(y_min, min(q3 + 0.5, y_max))
        plt.annotate(f'Q3: {q3:.2f}', xy=(i + 1, q3), xytext=(i + 1.1, q3_text_y),
                        fontsize=9, ha='left', va='center', arrowprops=dict(facecolor='green', arrowstyle='->'))

        plt.ylabel('Score Differences')
        plt.title(f'Boxplot of {prompt} Score Differences Between Versions - Annotation Method {aggregation}')
        plt.annotate(f'Number of Documents: {len(differences)}', xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        
        
        # Save boxplot
        plt.savefig(output_dir / (prompt + f'_score_distributions_difference_boxplot_{aggregation}.png'))
        plt.close()


def _prepare_df(document_scores: dict[str, dict[str, float]], aggregation: Union[None, str]) -> pd.DataFrame:
    """
    Prepares a DataFrame from the document scores dictionary, filtering out incomplete rows.

    Args:
        document_scores (dict[str, dict[str, float]]): A dictionary where keys are document IDs and 
                                                       values are dictionaries of version-to-score mappings.

    Returns:
        pd.DataFrame: A DataFrame with rows as document IDs and columns as versions, containing valid scores only.
    """
    if aggregation is not None:
        document_scores_for_df = document_scores
    else:
        # we add for each score a separate version
        document_scores_for_df = {}
        for doc_id, version_scores in document_scores.items():
            assert len(version_scores) == 1
            version = next(iter(version_scores.keys()))
            new_version_scores = {}
            for i, score in enumerate(version_scores[version]):
                new_version_scores[f"{version}_{i}"] = score
            document_scores_for_df[doc_id] = new_version_scores
            
    df = pd.DataFrame(document_scores_for_df).T  # Transpose for better structure (rows are IDs, columns are versions)
    df = df[sorted(df.columns)]  # Sort columns by version
    return df.dropna()  # Filter out documents not present in all versions
