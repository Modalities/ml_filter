import os
from pathlib import Path
import json
import pandas as pd

# Directory containing your JSON files
COMPARE_TO_GT_ONLY=True
input_directory = Path("/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/experiments/model_size_architecture/annotations/comparison_test")
output_excel = input_directory / "ir_summary_gt.xlsx"

# List to hold the extracted data
results = []

# Iterate through all JSON files in the input directory
for filename in os.listdir(input_directory):
    if filename.endswith(".json"):
        file_path = os.path.join(input_directory, filename)
        
        # Open and parse the JSON file
        with open(file_path, "r") as file:
            try:
                filename_without_ext = os.path.splitext(filename)[0]
                model1 = filename_without_ext.split("_")[-2]
                model2 = filename_without_ext.split("_")[-1]
                
                if COMPARE_TO_GT_ONLY and (model1 != "gt" and model2 != "gt"):
                    continue
                if model1 == model2:
                    continue
                
                # Extract values from the JSON structure
                content = json.load(file)       
                for prompt in content:
                    prompt_data = content.get(prompt, {})     
                    result = {}
                    if COMPARE_TO_GT_ONLY:
                        result["Model"] = model1 if model1 != "gt" else model2
                    else:
                        result["Model 1"] = model1
                        result["Model 2"] = model2
                    result["Prompt"] = prompt
                    result["Fleiss"] = prompt_data.get("Fleiss Kappa")
                    result["Cohen"] = prompt_data.get("Cohen Kappa (avg pairwise)")
                    result["Spearman"] = prompt_data.get("Spearman Rank Correlation (avg pairwise)")
                    result["Kendall"] = prompt_data.get("Kendall Tau (avg pairwise)")
                    result["Krippendorff"] = prompt_data.get("Krippendorff Alpha")
                    result["Filename"] = filename
                    results.append(result)
            
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON file {filename}: {e}")

# Convert the data to a DataFrame
df = pd.DataFrame(results)

# Write the DataFrame to an Excel file
df.to_excel(output_excel, index=False)

# Write the DataFrame to a LaTeX table
df = df.drop(columns=["Filename", "Prompt"])
columns_to_highlight_max = [col for col in df.columns if col not in ["Model", "Model 1", "Model 2", "Filename", "Prompt"]]
styled_df = df.style.highlight_max(axis=0, subset=columns_to_highlight_max, props='textbf:--rwrap;')
styled_df = styled_df.hide(axis='index')
latex_output = styled_df.to_latex()
print(latex_output)

print(f"Metrics successfully written to {output_excel}")
