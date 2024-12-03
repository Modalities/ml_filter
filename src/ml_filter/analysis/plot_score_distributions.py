
from itertools import combinations
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns


def plot_scores(path_to_files: list[str], output_dir: Path) -> None:
    document_scores = _get_document_scores(path_to_files)
    # iterate over different prompts
    for prompt in document_scores:
        df = _prepare_df(document_scores[prompt])

        # Plotting the distributions
        sns.set_theme(style='whitegrid')
        plt.figure(figsize=(10, 6))
        for model_name in df.columns:
            scores = df[model_name]
            # Create a histogram for each file's scores
            plt.hist(scores, bins=30, alpha=0.5, label=model_name, edgecolor='black')

        # Add labels and title
        plt.xlabel('Educational Score')
        plt.ylabel('Frequency')
        plt.title('Educational Score Distributions')

        # Place annotation below the legend in the upper right corner
        plt.legend(loc='upper right')
        plt.annotate(
            f'Number of Documents: {len(df)}',
            xy=(0.95, 0.85), xycoords='axes fraction',  # Adjust y-coordinate to be slightly below the legend
            fontsize=10, ha='right', va='top',
            bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white')
        )

        # Save and close the plot
        plt.savefig(output_dir / (prompt + '_score_distributions.png'))
        plt.close()



def plot_differences_in_scores(path_to_files: list[str], output_dir: Path) -> None:
    # Initialize a dictionary to store educational_score by id across files
    document_scores = _get_document_scores(path_to_files)
    
    # iterate over different prompts
    for prompt in document_scores:
        df = _prepare_df(document_scores[prompt])
        

        # Initialize a list to store the differences for each consecutive version pair
        score_differences = {}

        # Compute differences for consecutive versions for each document
        # TODO why only consecutive versions? Use arbitrary pairs of versions instead
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
            plt.title(f'Histogram of Educational Score Differences ({version2} - {version1})')
            plt.annotate(f'Number of Documents: {len(differences)}', xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

        plt.tight_layout()
        # Save histogram plot
        plt.savefig(output_dir / (prompt + '_score_distributions_difference_histogram.png'))
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
        plt.annotate(f'Q1: {q1:.2f}', xy=(i + 1, q1), xytext=(i + 1.1, q1 - 0.5),
                        fontsize=9, ha='left', va='center', arrowprops=dict(facecolor='blue', arrowstyle='->'))
        plt.annotate(f'Q3: {q3:.2f}', xy=(i + 1, q3), xytext=(i + 1.1, q3 + 0.5),
                        fontsize=9, ha='left', va='center', arrowprops=dict(facecolor='green', arrowstyle='->'))

        plt.ylabel('Score Differences')
        plt.title('Boxplot of Educational Score Differences Between Versions')
        plt.annotate(f'Number of Documents: {len(differences)}', xy=(0.95, 0.95), xycoords='axes fraction', 
                    fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))
        
        
        # Save boxplot
        plt.savefig(output_dir / (prompt + '_score_distributions_difference_boxplot.png'))
        plt.close()

def _get_document_scores(path_to_files: list[str]) -> dict[str, dict[str, float]]:
    document_scores = {}

    # Loop through each file
    for file_path in path_to_files:
        # Extract the first part of the filename for labeling (e.g., the version)
        prompt, prompt_lang, model = os.path.basename(file_path).split('_')[1:4]
        annotator_id = "_".join([model, prompt, prompt_lang])
        # Read the JSONL file and extract educational_score for each document
        with open(file_path, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                doc_id = json_obj.get('document_id')
                
                if not prompt in document_scores:
                    document_scores[prompt] = {}
                
                if doc_id not in document_scores[prompt]:
                    document_scores[prompt][doc_id] = {}
                        
                version = "_".join([prompt_lang, model])                    
                if version in document_scores[prompt][doc_id]:
                    raise ValueError(f"Found duplicate score for {annotator_id}")
                
                # aggregate scores
                # TODO add different types of aggregation
                scores = json_obj["scores"]
                aggr_score = min(scores)
                document_scores[prompt][doc_id][version] = aggr_score
    
    return document_scores

def _prepare_df(document_scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    # Convert the dictionary into a DataFrame for easier processing
    df = pd.DataFrame(document_scores).T  # Transpose for better structure (rows are ids, columns are versions)

    # Sort the columns to ensure the versions are in order
    df = df[sorted(df.columns)]

    # Filter documents that are available across all versions
    return df.dropna()
