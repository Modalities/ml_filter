from itertools import combinations
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import seaborn as sns
from ml_filter.analysis.utils import get_common_docs, get_document_scores


def plot_scores(path_to_files: Tuple[Path], output_dir: Path, labels: List[str], aggregation: Optional[str]) -> None:
    """
    Plots score distributions for each prompt based on the input JSONL files.

    Args:
        path_to_files (Tuple[Path]): A list of paths to JSONL files containing document scores.
        output_dir (Path): The directory to save the generated plots.
        aggregation (Optional[str]): Aggregation method for scores ("min", "max", "mean", "majority", or None).

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    document_scores = get_document_scores(
        path_to_files=path_to_files,
        aggregation=aggregation,
        labels=labels
    )
    
        # df = _prepare_df(document_scores[prompt])

    # Iterate over different prompts
    for prompt in document_scores["prompt"].unique():
        prompt_df = document_scores[document_scores["prompt"] == prompt]
        for annotator in prompt_df["model"].unique():
            annotator_df = prompt_df[prompt_df["model"] == annotator]

            # Plotting the distributions
            sns.set_theme(style='whitegrid')
            plt.figure(figsize=(10, 6))
            scores = annotator_df["score"]

            plt.hist(scores, bins=30, alpha=0.5, label=annotator, edgecolor='black')

            # Add labels, title, and annotations
            plt.xlabel(f'{prompt} Score')
            plt.ylabel('Frequency')
            plt.title(f'{prompt} Score Distributions - Annoator {annotator}')

            # Place annotation below the legend in the upper right corner
            plt.legend(loc='upper right')
            plt.annotate(
                f'Number of Documents: {len(document_scores)}',
                xy=(0.95, 0.85), xycoords='axes fraction',  # Adjust y-coordinate to be slightly below the legend
                fontsize=10, ha='right', va='top',
                bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
            )

            # Save and close the plot
            plt.savefig(output_dir / (prompt + f'_score_distributions_{annotator}.png'))
            plt.close()


def plot_differences_in_scores(path_to_files: Tuple[Path], output_dir: Path, labels: List[str], aggregation: Optional[str]) -> None:
    """
    Plots histograms and boxplots of score differences between different versions of models.

    Args:
        path_to_files (Tuple[Path]): A list of paths to JSONL files containing document scores.
        output_dir (Path): The directory to save the generated plots.
        aggregation (Optional[str]): Aggregation method for scores ("min", "max", "mean", "majority", or None).

    Returns:
        None
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    document_scores = get_document_scores(
        path_to_files=path_to_files,
        aggregation=aggregation,
        labels=labels
    )
    
    # Iterate over different prompts
    for prompt in document_scores["prompt"].unique():
        prompt_df = document_scores[document_scores["prompt"] == prompt]
        
        # Initialize a list to store the differences for each consecutive version pair
        score_differences = {}
            
        # Compute differences for each document for all pairs of versions 
        for (annotator_1, annotator_2) in combinations(prompt_df["model"].unique(), 2):
            # Compute the difference between the two versions for all documents that have scores in both versions
            common_docs_df = get_common_docs(prompt_df, annotator_1, annotator_2)
            score_differences[(annotator_1, annotator_2)] = (common_docs_df['rounded_score_1'] - common_docs_df['rounded_score_0']).to_list()

        sns.set_theme(style='whitegrid')
        # Plotting the differences as histograms where x-axis represents the difference and y-axis represents the frequency
        plt.figure(figsize=(12, 8))
        
        # Loop through the computed differences and plot histograms for each consecutive version pair
        for i, (annotator_1, annotator_2) in enumerate(score_differences):
            differences = score_differences[(annotator_1, annotator_2)]
            plt.subplot(len(score_differences), 1, i + 1)
            plt.hist(differences, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
            plt.xlabel('Score Difference')
            plt.ylabel('Frequency')
            plt.title(f'Histogram of {prompt} Score Differences ({annotator_2} - {annotator_1}) - Aggregation Method {aggregation}')
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

        plt.ylabel('Score Differences')
        plt.title(f'Boxplot of {prompt} Score Differences Between Versions - Aggregation Method {aggregation}')
        plt.annotate(f'Number of Documents: {len(differences)}', xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        
        # Rotate x-axis labels
        plt.xticks(rotation=90)
        plt.tight_layout()
        
        # Save boxplot
        plt.savefig(output_dir / (prompt + f'_score_distributions_difference_boxplot_{aggregation}.png'))
        plt.close()
