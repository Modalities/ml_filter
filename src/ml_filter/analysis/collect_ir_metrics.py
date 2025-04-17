from collections import defaultdict
import json
import logging
from pathlib import Path
import re

from matplotlib import pyplot as plt
import pandas as pd
from pandas.io.formats.style import Styler
import seaborn as sns

from ml_filter.utils.logging import get_logger

logger = get_logger(name=__name__, level=logging.INFO) # Set up logging


def style_df(
    df: pd.DataFrame,
    sort_key: str,
    max_columns: list[str],
    min_columns: list[str]
) -> Styler:
    """
    Styles a DataFrame by sorting it and highlighting the maximum and minimum values in specified columns.

    Args:
        df (pd.DataFrame): The DataFrame to style.
        sort_key (str): The column to sort the DataFrame by.
        max_columns (list[str]): Columns to highlight the maximum values.
        min_columns (list[str]): Columns to highlight the minimum values.

    Returns:
        Styler: A styled DataFrame.
    """
    df_sorted = df.sort_values(by=sort_key)
    # Ensure sort column is first and other columns are sorted alphabetically
    columns = [sort_key] + sorted([col for col in df.columns if col != sort_key and col in min_columns + max_columns])
    df_sorted = df_sorted[columns]
    styled_df = df_sorted.style.hide(axis='index')
    
    # highlight best values in each column
    styled_df = styled_df.highlight_max(axis=0, subset=max_columns, props='textbf:--rwrap;')
    styled_df = styled_df.highlight_min(axis=0, subset=min_columns, props='textbf:--rwrap;')
    return styled_df


def read_metric_data(input_directory: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Reads metric data from JSON files in the input directory and returns a DataFrame and a list of metrics.

    Example for returned dataframe:
    
    df.head():
        Fleiss     Cohen  Spearman  ...     Filepath                                                CM      lang
    0 -0.304884 -0.000976  0.102076  ...  /path/to/json  {'0': {'-1': 2, '0': 0, '1': 0, '2': 0, '3': 1...    bg
    1 -0.053517  0.009249  0.637778  ...  /path/to/json  {'0': {'-1': 0, '0': 11, '1': 71, '2': 66, '3'...    bg
    2  0.275123  0.282199  0.629632  ...  /path/to/json  {'0': {'-1': 0, '0': 94, '1': 60, '2': 17, '3'...    bg
    3 -0.094009 -0.015461  0.608285  ...  /path/to/json  {'0': {'-1': 0, '0': 0, '1': 83, '2': 65, '3':...    bg
    4  0.117063  0.147602  0.682103  ...  /path/to/json  {'0': {'-1': 22, '0': 65, '1': 51, '2': 35, '3...    bg

    Args:
        input_directory (Path): The directory containing the JSON files.

    Returns:
        tuple[pd.DataFrame, list[str]]: A DataFrame containing the metric data and a list of metrics.
    """
    metrics = set()
    data = list()
    
    # Iterate through all JSON files in the input directory
    for file_path in list(input_directory.rglob("ir_*.json")):     
        # Open and parse the JSON file
        with open(file_path, "r") as f:
            try:
                filename_without_ext = file_path.stem
                annotator_1 = filename_without_ext.split("_")[-2]
                annotator_2 = filename_without_ext.split("_")[-1]
                lang = file_path.parent.name
                
                # Extract values from the JSON structure
                content = json.load(f)       
                doc = {metric: value for metric, value in content["metrics"].items()} 
                metrics.update(doc.keys())
                if annotator_1 == "gt":
                    annotator = annotator_2
                elif annotator_2 == "gt":
                    annotator = annotator_1
                else:
                    raise ValueError(f"Expected one annotator to be 'gt' but got {annotator_1} and {annotator_2}")
                doc["Annotator"] = annotator
                doc["Filepath"] = file_path
                doc["CM"] = content.get("CM")
                doc["lang"] = lang
                data.append(doc)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON file {file_path}: {e}")
    df = pd.DataFrame(data)
    metrics = sorted(list(metrics))
    return df, metrics
    

def get_top_n_annotators(
    df: pd.DataFrame,
    top_n: int,
    max_metrics: list[str],
    min_metrics: list[str]
) -> dict[int, dict[str, dict[str, int]]]:
    """
    Gets the top n annotators for each metric.

    Args:
        df (pd.DataFrame): The DataFrame containing the metric data.
        top_n (int): The number of top annotators to select.
        max_metrics (list[str]): Metrics where higher values are better.
        min_metrics (list[str]): Metrics where lower values are better.

    Returns:
        dict[int, dict[str, dict[str, int]]]: A dictionary containing the top n annotators for each metric.
    """
    top_n_range = range(1, top_n + 1)
    top_n_annotators = {n: {metric: defaultdict(lambda: defaultdict(int)) for metric in max_metrics + min_metrics} for n in top_n_range}
    
    # get the top n models for each metric  
    for n in top_n_range:
        for lang in df["lang"].unique():
            lang_df = df[df["lang"] == lang]
            # initialize the dictionary with 0 for each annotator
            for annotator in lang_df["Annotator"].unique():
                for metric in max_metrics + min_metrics:
                    if annotator not in top_n_annotators[n][metric]: 
                        top_n_annotators[n][metric][annotator] = 0
            # count the number of times each annotator is in the top n
            for metric in max_metrics:
                for annotator in lang_df.nlargest(n, metric, keep="all")["Annotator"].to_list():
                    top_n_annotators[n][metric][annotator] += 1
            for metric in min_metrics:
                for annotator in lang_df.nsmallest(n, metric, keep="all")["Annotator"].to_list():
                    top_n_annotators[n][metric][annotator] += 1          

    return top_n_annotators


def write_latex_output(
    df: pd.DataFrame,
    aggregated_metrics_df: pd.DataFrame,
    top_n_annotators: dict[int, dict[str, dict[str, int]]],
    output_directory: Path,
    max_metrics: list[str],
    min_metrics: list[str]
) -> None:
    """
    Writes the metric data to a LaTeX table.

    Args:
        df (pd.DataFrame): The DataFrame containing the metric data.
        aggregated_metrics_df (pd.DataFrame): The DataFrame containing the aggregated metric data.
        top_n_annotators (dict[int, dict[str, dict[str, int]]]): A dictionary containing the top n annotators for each metric.
        output_directory (Path): The directory to save the LaTeX file.
        max_metrics (list[str]): Metrics where higher values are better.
        min_metrics (list[str]): Metrics where lower values are better.

    Returns:
        None
    """
    # Write the DataFrame to a LaTeX table
    latex_str = ""
    df = df.drop(columns=["Filepath", "CM"])
    for lang in df["lang"].unique():
        lang_df = df[df["lang"] == lang]
        styled_lang_df = style_df(
            df=lang_df,
            max_columns=max_metrics,
            min_columns=min_metrics,
            sort_key="Annotator"
        )
        latex_str += (
            f"""
            \\begin{{table}}[ht]
            \\centering
            \\scriptsize
            {styled_lang_df.to_latex()}
            \\caption{{Measures of agreement between LLM annotated and human annotated scores for language \\textbf{{{lang}}}}}
            \\label{{tab:llm_scores_{lang}}}
            \\end{{table}}
            """
        )
    # add annotators in index as a column
    aggregated_metrics_df = aggregated_metrics_df.reset_index().rename(columns={"index": "Annotator"})
    
    styled_aggregated_metrics_df = style_df(
        df=aggregated_metrics_df,
        max_columns=max_metrics,
        min_columns=min_metrics,
        sort_key="Annotator"
    )
    # add results aggregated over all languages
    latex_str += (
        f"""
        \\begin{{table}}[ht]
        \\centering
        \\scriptsize
        {styled_aggregated_metrics_df.to_latex()}
        \\caption{{Measures of agreement between LLM annotated and human annotated scores across languages}}
        \\label{{tab:llm_scores_all_langs}}
        \\end{{table}}
        """
    )
    # add tables for top n annotators
    for n in top_n_annotators:
        top_annotators_df = pd.DataFrame.from_dict(top_n_annotators[n])
        top_annotators_df = top_annotators_df.reset_index().rename(columns={"index": "Annotator"})
        styled_top_annotators_df = style_df(
            df=top_annotators_df,
            max_columns=max_metrics + min_metrics,
            min_columns=[],
            sort_key="Annotator"
        )
        latex_str += (
            f"""
            \\begin{{table}}[ht]
            \\centering
            \\scriptsize
            {styled_top_annotators_df.to_latex()}
            \\caption{{Number of times each LLM was under top {n} performing annotators across languages}}
            \\label{{tab:llm_top_n_all_langs}}
            \\end{{table}}
            """
        )
    
    # Replace newline characters followed by whitespaces with just a newline character
    latex_str = re.sub(r'\n\s+', '\n', latex_str)

    logging.info(f"Generated LaTeX tables:\n\n{latex_str}")
    with open(output_directory / f"ir_summary_gt.tex", "w") as f:
        f.write(latex_str)
          
          
def aggregate_across_languages(
    df: pd.DataFrame,
    metrics: list[str]
) -> tuple[pd.DataFrame, dict[str, dict[str, dict[str, int]]]]:
    """
    Aggregates the metric data across languages.

    Args:
        df (pd.DataFrame): The DataFrame containing the metric data.
        metrics (list[str]): The list of metrics to aggregate.

    Returns:
        tuple[pd.DataFrame, dict[str, dict[str, dict[str, int]]]]: A DataFrame containing the aggregated metric data and a dictionary containing the aggregated confusion matrices.
    """
    # Aggregate the values for each annotator across languages
    aggregated_cm = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    aggregated_metrics = defaultdict(lambda: defaultdict(float))
    for lang in df["lang"].unique():
        lang_df = df[df["lang"] == lang]
        for annotator in lang_df["Annotator"].unique():         
            annotator_df = lang_df[lang_df["Annotator"] == annotator]
            # aggregate metrics
            for metric in metrics:
                aggregated_metrics[annotator][metric] += annotator_df[metric].sum()
            aggregated_metrics[annotator]["Count"] += len(annotator_df)

            # aggregate confusion matrices
            if len(annotator_df["CM"]) != 1:
                raise ValueError(f"Expected exactly one confusion matrix for annotator {annotator} language {lang}")
            else:
                cm = list(annotator_df["CM"])[0]
                for label in cm:
                    for pred in cm[label]:
                        aggregated_cm[annotator][label][pred] += cm[label][pred]
                  
    aggregated_metrics_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')  
    # Divide the values in each row by the value in the column Count
    for metric in [m for m in metrics if m != "Invalid"]:
        aggregated_metrics_df[metric] = aggregated_metrics_df.apply(lambda row: row[metric] / row["Count"] if row["Count"] != 0 else 0, axis=1)
    aggregated_metrics_df = aggregated_metrics_df.drop(columns=["Count"])
    aggregated_metrics_df["Invalid"] = aggregated_metrics_df["Invalid"].astype(int)
    
    return aggregated_metrics_df, aggregated_cm


def plot_confusion_matrix(
    aggregated_cm: dict[str, dict[str, dict[str, int]]],
    output_directory: Path
) -> None:
    """
    Plots the confusion matrix for each annotator.

    Args:
        aggregated_cm (dict[str, dict[str, dict[str, int]]]): A dictionary containing the aggregated confusion matrices.
        output_directory (Path): The directory to save the confusion matrix plots.

    Returns:
        None
    """
    labels = sorted(set(str(label) for annotator in aggregated_cm for label in aggregated_cm[annotator]))
    predictions = sorted(set(str(pred) for annotator in aggregated_cm for label in aggregated_cm[annotator] for pred in aggregated_cm[annotator][label]))
    for annotator in aggregated_cm:
        # get the confusion matrix for the annotator and convert it to a list
        aggregated_cm_annotator = []
        for label in labels:
            preds_for_label = []
            for pred in predictions:
                preds_for_label.append(aggregated_cm[annotator][label].get(pred, 0))
            aggregated_cm_annotator.append(preds_for_label)
        
        # normalize the confusion matrix
        normalized_aggregated_cm_annotator = [[n/sum(preds) if sum(preds) > 0 else 0 for n in preds] for preds in aggregated_cm_annotator]
        
        # plot the confusion matrix
        plt.figure(figsize=(10, 6))
        xlabels = [p if p != "-1" else "invalid" for p in predictions]
        sns.heatmap(normalized_aggregated_cm_annotator, annot=True, fmt='.2f', cmap='Blues', xticklabels=xlabels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {annotator}')
        plt.savefig(output_directory / f"confusion_matrix_across_languages_{annotator}.png")

                
def plot_spearman_heatmap(df: pd.DataFrame, output_directory: Path) -> None:
    """
    Plots a heatmap for Spearman correlation values across annotators and languages.

    Args:
        df (pd.DataFrame): A DataFrame containing metrics for different annotators and languages.
        output_directory (Path): The directory to save the heatmap plot.

    Returns:
        None
    """
    # spearman_df = df.pivot(index="Annotator", columns="lang", values="Spearman")
    spearman_df = df.pivot(index="lang", columns="Annotator", values="Spearman")
    
    # Create the heatmap
    plt.figure(figsize=(12, 10))  # Increase figure size for better visibility
    sns.heatmap(
        spearman_df,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Spearman Correlation"},
        xticklabels=spearman_df.columns,
        yticklabels=spearman_df.index,
        annot_kws={"fontsize": 10},  # Adjust font size for annotations
    )
    plt.xticks(rotation=45, ha="right", fontsize=10)  # Rotate x-axis labels for better visibility
    plt.yticks(fontsize=10)  # Adjust font size for y-axis labels
    plt.title("Spearman Correlation Heatmap", fontsize=14)
    plt.xlabel("Annotator", fontsize=12)
    plt.ylabel("Language", fontsize=12)
    
    # Save the heatmap
    heatmap_path = output_directory / "spearman_correlation_heatmap.png"
    plt.tight_layout()  # Ensure everything fits within the figure
    plt.savefig(heatmap_path)
    plt.close()
    logging.info(f"Spearman correlation heatmap saved to {heatmap_path}")
    
                
def collect_ir_metrics(
    input_directory: Path,
    output_directory: Path,
    top_n: int = 4,
    min_metrics: list[str] | None = None,
) -> None:
    """
    Collects inter-rater reliability metrics and writes the results to plots and a LaTeX table.

    Args:
        input_directory (Path): The directory containing the input JSON files.
        output_directory (Path): The directory to save the LaTeX file and confusion matrix plots.
        top_n (int): The number of top annotators to select.
        min_metrics (list[str] or None, optional): Metrics where lower values are better.

    Returns:
        None
    """
    output_directory.mkdir(exist_ok=True)

    # Read the data from the input directory
    df, metrics = read_metric_data(input_directory=input_directory)

    if min_metrics is None:
        min_metrics = ["MAE", "MSE", "Invalid"]
    max_metrics = [metric for metric in metrics if metric not in min_metrics]
    
    # scores aggregated over all languages
    aggregated_metrics_df, aggregated_cm = aggregate_across_languages(df, metrics)
    top_n_annotators = get_top_n_annotators(df, top_n, max_metrics, min_metrics)
    
    # Write the results to a LaTeX table
    write_latex_output(df, aggregated_metrics_df, top_n_annotators, output_directory, max_metrics, min_metrics)
        
    # plot the confusion matrix
    plot_confusion_matrix(aggregated_cm=aggregated_cm, output_directory=output_directory)
    
    # plot heatmap for spearman correlation
    plot_spearman_heatmap(df, output_directory=output_directory)
    
    logging.info(f"Metrics successfully written")
