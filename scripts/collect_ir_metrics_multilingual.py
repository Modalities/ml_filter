
import json
from pathlib import Path
from typing import List

import pandas as pd
from pandas.io.formats.style import Styler


def style_df(df: pd.DataFrame, max_columns: List[str], min_columns: List[str]) -> Styler:
    df_sorted = df.sort_values(by="Model")
    styled_df = df_sorted.style.highlight_max(axis=0, subset=max_columns, props='textbf:--rwrap;')
    styled_df = styled_df.highlight_min(axis=0, subset=min_columns, props='textbf:--rwrap;')
    return styled_df.hide(axis='index')


# Directory containing your JSON files
COMPARE_TO_GT_ONLY=True
input_directory = Path("/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/experiments/multilinguality/experiments/comparison")
output_directory = input_directory / "ir_summary"

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
            
            if COMPARE_TO_GT_ONLY and (model1 != "gt" and model2 != "gt"):
                continue
            if model1 == model2:
                continue
            
            # Extract values from the JSON structure
            content = json.load(f)       
            for prompt in content:
                prompt_data = content.get(prompt, {})     
                result = {}
                result["Prompt"] = prompt
                if COMPARE_TO_GT_ONLY:
                    result["Model"] = model1 if model1 != "gt" else model2
                    result["Acc"] = prompt_data.get("Accuracy against GT (avg pairwise)")
                    result["MAE"] = prompt_data.get("MAE against GT (avg pairwise)")
                    result["MSE"] = prompt_data.get("MSE against GT (avg pairwise)")
                else:
                    result["Model 1"] = model1
                    result["Model 2"] = model2
                result["Fleiss"] = prompt_data.get("Fleiss Kappa")
                result["Cohen"] = prompt_data.get("Cohen Kappa (avg pairwise)")
                result["Spearman"] = prompt_data.get("Spearman Rank Correlation (avg pairwise)")
                result["Kendall"] = prompt_data.get("Kendall Tau (avg pairwise)")
                result["Krippendorff"] = prompt_data.get("Krippendorff Alpha")
                result["Invalid"] = prompt_data.get("Number of invalid scores", 0)
                result["Filepath"] = file_path
                if not lang in results:
                    results[lang] = []
                results[lang].append(result)
                
        
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON file {file_path}: {e}")

latex_output = ""
# Initialize a DataFrame to hold the aggregated values for each model
aggregated_results = pd.DataFrame(columns=["Model", "Acc", "MAE", "MSE", "Fleiss", "Cohen", "Spearman", "Kendall", "Krippendorff", "Invalid", "Count"])
min_columns = ["MAE", "MSE", "Invalid"]
max_columns = [col for col in aggregated_results.columns if col not in (["Model", "Count"] + min_columns)]
top_n_models = {n: {col: {} for col in min_columns + max_columns} for n in [1, 2, 3, 4]}

for lang in sorted(results.keys()):
    # Convert the data to a DataFrame
    df = pd.DataFrame(results[lang])
    
    # get the top n models for each metric
    for col in max_columns + min_columns:
        for model in df["Model"].unique():
            for n in top_n_models.keys():
                if model not in top_n_models[n][col]:
                    top_n_models[n][col][model] = 0
        
    for col in max_columns:
        for n in top_n_models.keys():
            for model in df.nlargest(n, col)["Model"].to_list():
                top_n_models[n][col][model] += 1

    for col in min_columns:
        for n in top_n_models.keys():
            for model in df.nsmallest(n, col)["Model"].to_list():
                top_n_models[n][col][model] += 1

    # Aggregate the values for each model
    for model in df["Model"].unique():
        model_df = df[df["Model"] == model]
        if model not in aggregated_results["Model"].values:
            new_row = pd.DataFrame([{
                "Model": model,
                "Acc": 0,
                "MAE": 0,
                "MSE": 0,
                "Fleiss": 0,
                "Cohen": 0,
                "Spearman": 0,
                "Kendall": 0,
                "Krippendorff": 0,
                "Invalid": 0,
                "Count": 0
            }])
            aggregated_results = pd.concat([aggregated_results, new_row], ignore_index=True)

        aggregated_results.loc[aggregated_results["Model"] == model, "Acc"] += model_df["Acc"].sum()
        aggregated_results.loc[aggregated_results["Model"] == model, "MAE"] += model_df["MAE"].sum()
        aggregated_results.loc[aggregated_results["Model"] == model, "MSE"] += model_df["MSE"].sum()
        aggregated_results.loc[aggregated_results["Model"] == model, "Fleiss"] += model_df["Fleiss"].sum()
        aggregated_results.loc[aggregated_results["Model"] == model, "Cohen"] += model_df["Cohen"].sum()
        aggregated_results.loc[aggregated_results["Model"] == model, "Spearman"] += model_df["Spearman"].sum()
        aggregated_results.loc[aggregated_results["Model"] == model, "Kendall"] += model_df["Kendall"].sum()
        aggregated_results.loc[aggregated_results["Model"] == model, "Krippendorff"] += model_df["Krippendorff"].sum()
        aggregated_results.loc[aggregated_results["Model"] == model, "Invalid"] += model_df["Invalid"].sum()
        aggregated_results.loc[aggregated_results["Model"] == model, "Count"] += len(model_df)

    # Write the DataFrame to an Excel file
    df.to_excel(output_directory / f"ir_summary_gt_{lang}.xlsx", index=False)

    # Write the DataFrame to a LaTeX table
    df = df.drop(columns=["Filepath", "Prompt"])
    df["Invalid"] = df["Invalid"].astype(int)
    styled_df = style_df(
        df=df,
        max_columns=max_columns,
        min_columns=min_columns
    )
    latex_output += f"""
\\begin{{table}}[ht]
\\centering
\\scriptsize
{styled_df.to_latex()}
\\caption{{Measures of agreement between LLM annotated and human annotated scores for language \\textbf{{{lang}}}}}
\\label{{tab:llm_scores_{lang}}}
\\end{{table}}
"""

# scores aggregated over all languages
# Divide the values in each row by the value in the column Count
for col in ["Acc", "MAE", "MSE", "Fleiss", "Cohen", "Spearman", "Kendall", "Krippendorff"]:
    aggregated_results[col] = aggregated_results.apply(lambda row: row[col] / row["Count"] if row["Count"] != 0 else 0, axis=1)
aggregated_results = aggregated_results.drop(columns=["Count"])
aggregated_results["Invalid"] = aggregated_results["Invalid"].astype(int)

# Write the DataFrame to an Excel file
aggregated_results.to_excel(output_directory / f"ir_summary_gt_all_langs.xlsx", index=False)

# add to latex output
styled_aggregated_results = style_df(
    df=aggregated_results,
    max_columns=max_columns,
    min_columns=min_columns
)
latex_output += f"""
\\begin{{table}}[ht]
\\centering
\\scriptsize
{styled_aggregated_results.to_latex()}
\\caption{{Measures of agreement between LLM annotated and human annotated scores across languages}}
\\label{{tab:llm_scores_all_langs}}
\\end{{table}}
"""

# Add top n models to latex output
top_n_models_dfs = {}
for n, metrics_dict in top_n_models.items():
    df = pd.DataFrame(metrics_dict).reset_index().rename(columns={"index": "Model"})
    styled_df = style_df(
        df=df,
        max_columns=[],
        min_columns=[],
    )
    latex_output += f"""
\\begin{{table}}[ht]
\\centering
\\scriptsize
{styled_df.to_latex()}
\\caption{{Number of times each LLM was under top {n} performing models across languages}}
\\label{{tab:llm_top_n_all_langs}}
\\end{{table}}
"""

print(latex_output)
with open(output_directory / "ir_summary_gt.tex", "w") as f:
    f.write(latex_output)

print(f"Metrics successfully written")
