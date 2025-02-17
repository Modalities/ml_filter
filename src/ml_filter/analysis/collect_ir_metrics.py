from collections import defaultdict
import json
from pathlib import Path
import re
from typing import List, Optional

from matplotlib import pyplot as plt
import pandas as pd
from pandas.io.formats.style import Styler
import seaborn as sns


def style_df(df: pd.DataFrame, sort_key: str, max_columns: Optional[List[str]] = None, min_columns: Optional[List[str]] = None) -> Styler:
    df_sorted = df.sort_values(by=sort_key)
    styled_df = df_sorted.style.hide(axis='index')
    if max_columns is not None:
        styled_df = styled_df.highlight_max(axis=0, subset=max_columns, props='textbf:--rwrap;')
    if min_columns is not None:
        styled_df = styled_df.highlight_min(axis=0, subset=min_columns, props='textbf:--rwrap;')
    return styled_df


def collect_ir_metrics(
    input_directory: Path,
    output_directory: Path,
    top_n: int = 4,
) -> None:
    output_directory.mkdir(exist_ok=True)

    # Initialize variables to store the results
    results = defaultdict(lambda: defaultdict(list))
    latex_output = ""
    aggregated_results = defaultdict(lambda: defaultdict(float))
    aggregated_cm = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    metrics = set()

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
                for prompt in content:
                    prompt_data = content.get(prompt, {})  
                    result = {metric: value for metric, value in prompt_data["metrics"].items()} 
                    metrics.update(result.keys())  
                    result["Model"] = model1 if model1 != "gt" else model2
                    result["Filepath"] = file_path
                    result["CM"] = prompt_data.get("CM")
                    results[prompt][lang].append(result)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {file_path}: {e}")

    metrics = sorted(list(metrics))
    min_metrics = ["MAE", "MSE", "Invalid"]
    max_metrics = [metric for metric in metrics if metric not in min_metrics]
    top_n_range = range(1, top_n + 1)
    top_n_models = {n: {metric: defaultdict(lambda: defaultdict(int)) for metric in metrics} for n in top_n_range}
    
    for prompt in results:
        for lang in sorted(results[prompt]):
            # Convert the data to a DataFrame
            metrics_df = pd.DataFrame(results[prompt][lang])
            
            # get the top n models for each metric  
            for n in top_n_range:
                # initialize the dictionary with 0 for each model
                for model in metrics_df["Model"].unique():
                    for metric in metrics:
                        if model not in top_n_models[n][metric]: 
                            top_n_models[n][metric][model] = 0
                # count the number of times each model is in the top n
                for metric in max_metrics:
                    for model in metrics_df.nlargest(n, metric)["Model"].to_list():
                        top_n_models[n][metric][model] += 1
                for metric in min_metrics:
                    for model in metrics_df.nsmallest(n, metric)["Model"].to_list():
                        top_n_models[n][metric][model] += 1          

            # Aggregate the values for each model            
            for model in metrics_df["Model"].unique():
                model_df = metrics_df[metrics_df["Model"] == model]
                for metric in metrics:
                    aggregated_results[model][metric] += model_df[metric].sum()
                aggregated_results[model]["Count"] += len(model_df)

                if len(model_df["CM"]) != 1:
                    raise ValueError(f"Expected exactly one confusion matrix for model {model} in prompt {prompt} and language {lang}")
                else:
                    cm = list(model_df["CM"])[0]
                    for label in cm:
                        for pred in cm[label]:
                            aggregated_cm[model][label][pred] += cm[label][pred]

            # Write the DataFrame to an Excel file
            metrics_df.to_excel(output_directory / f"ir_summary_{prompt}_gt_{lang}.xlsx", index=False)

            # Write the DataFrame to a LaTeX table
            metrics_df = metrics_df.drop(columns=["Filepath", "CM"])
            metrics_df["Invalid"] = metrics_df["Invalid"].astype(int) # TODO is this necessary?
            styled_metrics_df = style_df(
                df=metrics_df,
                max_columns=max_metrics,
                min_columns=min_metrics,
                sort_key="Model"
            )
            latex_output += (
                f"""
                \\begin{{table}}[ht]
                \\centering
                \\scriptsize
                {styled_metrics_df.to_latex()}
                \\caption{{Measures of agreement between LLM annotated and human annotated scores for prompt {prompt} and language \\textbf{{{lang}}}}}
                \\label{{tab:llm_scores_{prompt}_{lang}}}
                \\end{{table}}
                """
            )

        # scores aggregated over all languages
        aggregated_results_df = pd.DataFrame.from_dict(aggregated_results, orient='index')
        
        # Divide the values in each row by the value in the column Count
        for metric in [m for m in metrics if m != "Invalid"]:
            aggregated_results_df[metric] = aggregated_results_df.apply(lambda row: row[metric] / row["Count"] if row["Count"] != 0 else 0, axis=1)
        aggregated_results_df = aggregated_results_df.drop(columns=["Count"])
        aggregated_results_df["Invalid"] = aggregated_results_df["Invalid"].astype(int)

        # convert index to column
        aggregated_results_df = aggregated_results_df.reset_index().rename(columns={"index": "Model"})
        
        # Write the DataFrame to an Excel file
        aggregated_results_df.to_excel(output_directory / f"ir_summary_{prompt}_gt_all_langs.xlsx", index=False)

        # add to latex output
        styled_aggregated_results_df = style_df(
            df=aggregated_results_df,
            max_columns=max_metrics,
            min_columns=min_metrics,
            sort_key="Model"
        )
        latex_output += (
            f"""
            \\begin{{table}}[ht]
            \\centering
            \\scriptsize
            {styled_aggregated_results_df.to_latex()}
            \\caption{{Measures of agreement between LLM annotated and human annotated scores for prompt {prompt} across languages}}
            \\label{{tab:llm_scores_{prompt}_all_langs}}
            \\end{{table}}
            """
        )

        # Add top n models to latex output
        for n, metrics_dict in top_n_models.items():
            metrics_df = pd.DataFrame(metrics_dict).reset_index().rename(columns={"index": "Model"})
            styled_metrics_df = style_df(
                df=metrics_df,
                max_columns=metrics,
                sort_key="Model"
            )
            latex_output += (
                f"""
                \\begin{{table}}[ht]
                \\centering
                \\scriptsize
                {styled_metrics_df.to_latex()}
                \\caption{{Number of times each LLM was under top {n} performing models for prompt {prompt} across languages}}
                \\label{{tab:llm_top_n_{prompt}_all_langs}}
                \\end{{table}}
                """
            )

        # Replace newline characters followed by whitespaces with just a newline character
        latex_output = re.sub(r'\n\s+', '\n', latex_output)

        print(latex_output)
        with open(output_directory / f"ir_summary_{prompt}_gt.tex", "w") as f:
            f.write(latex_output)
            
        # Plot the aggregated confusion matrix
        labels = sorted(set(str(label) for model in aggregated_cm for label in aggregated_cm[model]))
        predictions = sorted(set(str(pred) for model in aggregated_cm for label in aggregated_cm[model] for pred in aggregated_cm[model][label]))
        for model in aggregated_cm:
            aggregated_cm_list = []
            for label in labels:
                label_list = []
                for pred in predictions:
                    label_list.append(aggregated_cm[model][label].get(pred, 0))
                aggregated_cm_list.append(label_list)
            
            normalized_aggregated_cm_list = [[n/sum(preds) if sum(preds) > 0 else 0 for n in preds] for preds in aggregated_cm_list]
            plt.figure(figsize=(10, 6))
            xlabels = [p if p != "-1" else "invalid" for p in predictions]
            sns.heatmap(normalized_aggregated_cm_list, annot=True, fmt='.2f', cmap='Blues', xticklabels=xlabels, yticklabels=labels)
            plt.xlabel('Predicted')
            plt.ylabel('True')
            plt.title(f'Confusion Matrix for {model}')
            plt.savefig(output_directory / f"confusion_matrix_{prompt}_across_languages_{model}.png")

    print(f"Metrics successfully written")
