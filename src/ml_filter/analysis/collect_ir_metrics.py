from collections import defaultdict
import json
import logging
from pathlib import Path
import re
from typing import List, Optional

from matplotlib import pyplot as plt
import pandas as pd
from pandas.io.formats.style import Styler
import seaborn as sns

from ml_filter.utils.logging import get_logger

logger = get_logger(name=__name__, level=logging.INFO) # Set up logging


def style_df(df: pd.DataFrame, sort_key: str, max_columns: Optional[List[str]] = None, min_columns: Optional[List[str]] = None) -> Styler:
    df_sorted = df.sort_values(by=sort_key)
    styled_df = df_sorted.style.hide(axis='index')
    if max_columns is not None:
        styled_df = styled_df.highlight_max(axis=0, subset=max_columns, props='textbf:--rwrap;')
    if min_columns is not None:
        styled_df = styled_df.highlight_min(axis=0, subset=min_columns, props='textbf:--rwrap;')
    return styled_df


def read_metric_data(input_directory: Path) -> None:
    metrics = set()
    data = list()
    
    # Iterate through all JSON files in the input directory
    for file_path in list(input_directory.rglob("ir_*.json")):     
        # Open and parse the JSON file
        with open(file_path, "r") as f:
            try:
                filename_without_ext = file_path.stem
                model1 = filename_without_ext.split("_")[-2]
                model2 = filename_without_ext.split("_")[-1]
                lang = file_path.parent.name
                
                # Extract values from the JSON structure
                content = json.load(f)       
                doc = {metric: value for metric, value in content["metrics"].items()} 
                metrics.update(doc.keys())  
                doc["Model"] = model1 if model1 != "gt" else model2
                doc["Filepath"] = file_path
                doc["CM"] = content.get("CM")
                doc["lang"] = lang
                data.append(doc)
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON file {file_path}: {e}")
    df = pd.DataFrame(data)
    metrics = sorted(list(metrics))
    return df, metrics
    

def get_top_n_models(df, top_n, max_metrics, min_metrics):
    top_n_range = range(1, top_n + 1)
    top_n_models = {n: {metric: defaultdict(lambda: defaultdict(int)) for metric in max_metrics + min_metrics} for n in top_n_range}
    
    # get the top n models for each metric  
    for n in top_n_range:
        for lang in df["lang"].unique():
            lang_df = df[df["lang"] == lang]
            # initialize the dictionary with 0 for each model
            for model in lang_df["Model"].unique():
                for metric in max_metrics + min_metrics:
                    if model not in top_n_models[n][metric]: 
                        top_n_models[n][metric][model] = 0
            # count the number of times each model is in the top n
            for metric in max_metrics:
                for model in lang_df.nlargest(n, metric)["Model"].to_list():
                    top_n_models[n][metric][model] += 1
            for metric in min_metrics:
                for model in lang_df.nsmallest(n, metric)["Model"].to_list():
                    top_n_models[n][metric][model] += 1          

    return top_n_models


def write_latex_output(df, aggregated_metrics_df, top_n_models, output_directory, max_metrics, min_metrics):
    # Write the DataFrame to a LaTeX table
    latex_str = ""
    df = df.drop(columns=["Filepath", "CM"])
    for lang in df["lang"].unique():
        lang_df = df[df["lang"] == lang]
        styled_lang_df = style_df(
            df=lang_df,
            max_columns=max_metrics,
            min_columns=min_metrics,
            sort_key="Model"
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
    # add models in index as a column
    aggregated_metrics_df = aggregated_metrics_df.reset_index().rename(columns={"index": "Model"})
    
    styled_aggregated_metrics_df = style_df(
        df=aggregated_metrics_df,
        max_columns=max_metrics,
        min_columns=min_metrics,
        sort_key="Model"
    )
    # results aggregated over all languages
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
    for n in top_n_models:
        top_models_df = pd.DataFrame.from_dict(top_n_models[n])
        top_models_df = top_models_df.reset_index().rename(columns={"index": "Model"})
        styled_top_models_df = style_df(
            df=top_models_df,
            max_columns=max_metrics,
            min_columns=min_metrics,
            sort_key="Model"
        )
        # top n models
        latex_str += (
            f"""
            \\begin{{table}}[ht]
            \\centering
            \\scriptsize
            {styled_top_models_df.to_latex()}
            \\caption{{Number of times each LLM was under top {n} performing models across languages}}
            \\label{{tab:llm_top_n_all_langs}}
            \\end{{table}}
            """
        )
    
    # Replace newline characters followed by whitespaces with just a newline character
    latex_str = re.sub(r'\n\s+', '\n', latex_str)

    logging.info(f"Generated LaTeX tables:\n\n{latex_str}")
    with open(output_directory / f"ir_summary_gt.tex", "w") as f:
        f.write(latex_str)
          
          
def aggregate_across_languages(df, metrics):
    # Aggregate the values for each model across languages
    aggregated_cm = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    aggregated_metrics = defaultdict(lambda: defaultdict(float))
    for lang in df["lang"].unique():
        lang_df = df[df["lang"] == lang]
        for model in lang_df["Model"].unique():         
            model_df = lang_df[lang_df["Model"] == model]
            # aggregate metrics
            for metric in metrics:
                aggregated_metrics[model][metric] += model_df[metric].sum()
            aggregated_metrics[model]["Count"] += len(model_df)

            # aggregate confusion matrices
            if len(model_df["CM"]) != 1:
                raise ValueError(f"Expected exactly one confusion matrix for model {model} language {lang}")
            else:
                cm = list(model_df["CM"])[0]
                for label in cm:
                    for pred in cm[label]:
                        aggregated_cm[model][label][pred] += cm[label][pred]
                  
    aggregated_metrics_df = pd.DataFrame.from_dict(aggregated_metrics, orient='index')  
    # Divide the values in each row by the value in the column Count
    for metric in [m for m in metrics if m != "Invalid"]:
        aggregated_metrics_df[metric] = aggregated_metrics_df.apply(lambda row: row[metric] / row["Count"] if row["Count"] != 0 else 0, axis=1)
    aggregated_metrics_df = aggregated_metrics_df.drop(columns=["Count"])
    aggregated_metrics_df["Invalid"] = aggregated_metrics_df["Invalid"].astype(int)
    
    return aggregated_metrics_df, aggregated_cm


def plot_confusion_matrix(aggregated_cm, output_directory):
    labels = sorted(set(str(label) for model in aggregated_cm for label in aggregated_cm[model]))
    predictions = sorted(set(str(pred) for model in aggregated_cm for label in aggregated_cm[model] for pred in aggregated_cm[model][label]))
    for model in aggregated_cm:
        # get the confusion matrix for the model and convert it to a list
        aggregated_cm_model = []
        for label in labels:
            preds_for_label = []
            for pred in predictions:
                preds_for_label.append(aggregated_cm[model][label].get(pred, 0))
            aggregated_cm_model.append(preds_for_label)
        
        # normalize the confusion matrix
        normalized_aggregated_cm_model = [[n/sum(preds) if sum(preds) > 0 else 0 for n in preds] for preds in aggregated_cm_model]
        
        # plot the confusion matrix
        plt.figure(figsize=(10, 6))
        xlabels = [p if p != "-1" else "invalid" for p in predictions]
        sns.heatmap(normalized_aggregated_cm_model, annot=True, fmt='.2f', cmap='Blues', xticklabels=xlabels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {model}')
        plt.savefig(output_directory / f"confusion_matrix_across_languages_{model}.png")

                
def collect_ir_metrics(
    input_directory: Path,
    output_directory: Path,
    top_n: int = 4,
    min_metrics: Optional[List[str]] = None, # TODO add to calling function and click command
) -> None:
    output_directory.mkdir(exist_ok=True)

    # Read the data from the input directory
    df, metrics = read_metric_data(input_directory=input_directory)

    if min_metrics is None:
        min_metrics = ["MAE", "MSE", "Invalid"]
    max_metrics = [metric for metric in metrics if metric not in min_metrics]
    
    # scores aggregated over all languages
    aggregated_metrics_df, aggregated_cm = aggregate_across_languages(df, metrics)
    top_n_models = get_top_n_models(df, top_n, max_metrics, min_metrics)
    
    # Write the results to a LaTeX table
    write_latex_output(df, aggregated_metrics_df, top_n_models, output_directory, max_metrics, min_metrics)
        
    # plot the confusion matrix
    plot_confusion_matrix(aggregated_cm=aggregated_cm, output_directory=output_directory)
    
    logging.info(f"Metrics successfully written")
