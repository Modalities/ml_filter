from collections import defaultdict
import json
from pathlib import Path
from typing import List

from matplotlib import pyplot as plt
import pandas as pd
from pandas.io.formats.style import Styler
import seaborn as sns


def style_df(df: pd.DataFrame, max_columns: List[str], min_columns: List[str]) -> Styler:
    df_sorted = df.sort_values(by="Model")
    styled_df = df_sorted.style.highlight_max(axis=0, subset=max_columns, props='textbf:--rwrap;')
    styled_df = styled_df.highlight_min(axis=0, subset=min_columns, props='textbf:--rwrap;')
    return styled_df.hide(axis='index')


def collect_ir_metrics(
    input_directory: Path,
    output_directory: Path
):
    output_directory.mkdir(exist_ok=True)

    # List to hold the extracted data
    results = {}

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
                    result = {
                        "Prompt": prompt,
                        "Model": model1 if model1 != "gt" else model2,
                        "Acc": prompt_data.get("Accuracy against GT (avg pairwise)"),
                        "MAE": prompt_data.get("MAE against GT (avg pairwise)"),
                        "MSE": prompt_data.get("MSE against GT (avg pairwise)"),
                        "Fleiss": prompt_data.get("Fleiss Kappa"),
                        "Cohen": prompt_data.get("Cohen Kappa (avg pairwise)"),
                        "Spearman": prompt_data.get("Spearman Rank Correlation (avg pairwise)"),
                        "Kendall": prompt_data.get("Kendall Tau (avg pairwise)"),
                        "Krippendorff": prompt_data.get("Krippendorff Alpha"),
                        "Invalid": prompt_data.get("Number of invalid scores", 0),
                        "Filepath": file_path,
                        "CM": prompt_data.get("CM against GT")
                    }
                    if not lang in results:
                        results[lang] = []
                    results[lang].append(result)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {file_path}: {e}")

    latex_output = ""
    aggregated_results = defaultdict(lambda: defaultdict(float))
    aggregated_cm = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    metrics = ["Acc", "MAE", "MSE", "Fleiss", "Cohen", "Spearman", "Kendall", "Krippendorff", "Invalid"]
    min_metrics = ["MAE", "MSE", "Invalid"]
    max_metrics = [metric for metric in metrics if metric not in min_metrics]
    top_n = range(1, 5)
    top_n_models = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for lang in sorted(results.keys()):
        # Convert the data to a DataFrame
        metrics_df = pd.DataFrame(results[lang])
        
        # get the top n models for each metric  
        for n in top_n:
            for metric in max_metrics:
                for model in metrics_df.nlargest(n, metric)["Model"].to_list():
                    top_n_models[n][metric][model] += 1
            for metric in min_metrics:
                for model in metrics_df.nsmallest(n, metric)["Model"].to_list():
                    top_n_models[n][metric][model] += 1          

        # Aggregate the values for each model
        for model in metrics_df["Model"].unique():
            model_df = metrics_df[metrics_df["Model"] == model]
            if model not in aggregated_results:
                aggregated_results[model]["Model"] = model
                aggregated_results[model]["Count"] = 0

            for metric in metrics:
                aggregated_results[model][metric] += model_df[metric].sum()
            aggregated_results[model]["Count"] += len(model_df)

            cm = list(model_df["CM"])[0]
            for label in cm:
                for pred in cm[label]:
                    aggregated_cm[model][label][pred] += cm[label][pred]

        # Write the DataFrame to an Excel file
        metrics_df.to_excel(output_directory / f"ir_summary_gt_{lang}.xlsx", index=False)

        # Write the DataFrame to a LaTeX table
        metrics_df = metrics_df.drop(columns=["Filepath", "Prompt", "CM"])
        metrics_df["Invalid"] = metrics_df["Invalid"].astype(int)
        styled_metrics_df = style_df(
            df=metrics_df,
            max_columns=max_metrics,
            min_columns=min_metrics
        )
        latex_output += (
            f"""
            \\begin{{table}}[ht]
            \\centering
            \\scriptsize
            {styled_metrics_df.to_latex()}
            \\caption{{Measures of agreement between LLM annotated and human annotated scores for language \\textbf{{{lang}}}}}
            \\label{{tab:llm_scores_{lang}}}
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

    # Write the DataFrame to an Excel file
    aggregated_results_df.to_excel(output_directory / f"ir_summary_gt_all_langs.xlsx", index=False)

    # add to latex output
    styled_aggregated_results_df = style_df(
        df=aggregated_results_df,
        max_columns=max_metrics,
        min_columns=min_metrics
    )
    latex_output += (
        f"""
        \\begin{{table}}[ht]
        \\centering
        \\scriptsize
        {styled_aggregated_results_df.to_latex()}
        \\caption{{Measures of agreement between LLM annotated and human annotated scores across languages}}
        \\label{{tab:llm_scores_all_langs}}
        \\end{{table}}
        """
    )

    # Add top n models to latex output
    for n, metrics_dict in top_n_models.items():
        metrics_df = pd.DataFrame(metrics_dict).reset_index().rename(columns={"index": "Model"})
        styled_metrics_df = style_df(df=metrics_df)
        latex_output += (
            f"""
            \\begin{{table}}[ht]
            \\centering
            \\scriptsize
            {styled_metrics_df.to_latex()}
            \\caption{{Number of times each LLM was under top {n} performing models across languages}}
            \\label{{tab:llm_top_n_all_langs}}
            \\end{{table}}
            """
        )

    print(latex_output)
    with open(output_directory / "ir_summary_gt.tex", "w") as f:
        f.write(latex_output)
        
    # Plot the aggregated confusion matrix
    labels = ["0", "1", "2", "3", "4", "5"]
    predictions = labels + ["-1"]
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
        plt.savefig(output_directory / f"confusion_matrix_across_languages_{model}.png")

    print(f"Metrics successfully written")
