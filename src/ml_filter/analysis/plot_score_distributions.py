import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_scores(path_to_files: list[str], output_dir: str) -> None:
    document_scores = _get_document_scores(path_to_files)
    df = _prepare_df(document_scores)

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
    plt.savefig(os.path.join(output_dir, 'score_distributions.png'))
    plt.close()



def plot_scores_differences(path_to_files: list[str], output_dir: str) -> None:
    # Initialize a dictionary to store educational_score by id across files
    document_scores = _get_document_scores(path_to_files)
    df = _prepare_df(document_scores)
    

    # Initialize a list to store the differences for each consecutive version pair
    score_differences = []

    # Compute differences for consecutive versions for each document
    for i in range(len(df.columns) - 1):
        version1 = df.columns[i]
        version2 = df.columns[i + 1]
    
        # Compute the difference between the two versions for all documents that have scores in both versions
        differences = df[version2] - df[version1]
        score_differences.append(differences)

    sns.set_theme(style='whitegrid')
    # Plotting the differences as histograms where x-axis represents the difference and y-axis represents the frequency
    plt.figure(figsize=(12, 8))
    
    # Loop through the computed differences and plot histograms for each consecutive version pair
    for i, differences in enumerate(score_differences):
        plt.subplot(len(score_differences), 1, i + 1)
        plt.hist(differences, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Score Difference')
        plt.ylabel('Frequency')
        plt.title(f'Histogram of Educational Score Differences ({df.columns[i]} to {df.columns[i + 1]})')
        plt.annotate(f'Number of Documents: {len(differences)}', xy=(0.95, 0.95), xycoords='axes fraction', 
                 fontsize=10, ha='right', va='top', bbox=dict(boxstyle='round,pad=0.3', edgecolor='black', facecolor='white'))

    plt.tight_layout()
    # Save histogram plot
    plt.savefig(os.path.join(output_dir, 'score_distributions_difference_histogram.png'))
    plt.close()

    # Plot boxplot of the score differences
    sns.reset_defaults()
    plt.figure(figsize=(12, 8))
    plt.boxplot(score_differences, labels=[f'{df.columns[i]} to {df.columns[i + 1]}' for i in range(len(df.columns) - 1)], showmeans=True, meanprops=dict(marker='x', markerfacecolor='red', markeredgecolor='red'))
    
    plt.figure(figsize=(10, 6))
    plt.boxplot(score_differences, labels=[f'{df.columns[i]} to {df.columns[i + 1]}' for i in range(len(df.columns) - 1)], showmeans=True)
    mean_value = np.mean(score_differences)
    median_value = np.median(score_differences)
    q1 = np.percentile(differences, 25)
    q3 = np.percentile(score_differences, 75)

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
    plt.savefig(os.path.join(output_dir, 'score_distributions_difference_boxplot.png'))
    plt.close()

def _get_document_scores(path_to_files: list[str]) -> dict[str, dict[str, float]]:
    document_scores = {}

    # Loop through each file
    for file_path in path_to_files:
        # Extract the first part of the filename for labeling (e.g., the version)
        version = os.path.basename(file_path).split('_')[0]
    
        # Read the JSONL file and extract educational_score for each document
        with open(file_path, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                doc_id = json_obj.get('id')
                if 'educational_score' in json_obj:
                    score = json_obj['educational_score']
                
                    # Store the score by document id and version
                    if doc_id not in document_scores:
                        document_scores[doc_id] = {}
                    document_scores[doc_id][version] = score
    
    return document_scores

def _prepare_df(document_scores: dict[str, dict[str, float]]) -> pd.DataFrame:
    # Convert the dictionary into a DataFrame for easier processing
    df = pd.DataFrame(document_scores).T  # Transpose for better structure (rows are ids, columns are versions)

    # Sort the columns to ensure the versions are in order
    df = df[sorted(df.columns)]

    # Filter documents that are available across all versions
    return df.dropna()
