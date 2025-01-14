from pathlib import Path
from ml_filter.analysis.interrater_reliability import compute_interrater_reliability_metrics


#### Parameters ####
input_directory = Path("/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/experiments/model_size_architecture/annotations")
output_directory = input_directory / "comparison_test"
subfolder_per_language = False
compare_to_gt_only = True
aggregation = "majority"
####################


# Find all files matching the pattern
files = list(input_directory.glob("annotations_*.jsonl"))

# Check if there are at least two files
if len(files) < 2:
    print("Not enough files to create tuples. Exiting.")
    exit(1)

output_directory.mkdir(parents=True, exist_ok=True)

# Function to extract the model name from the filename
def extract_model_name(filename):
    basename = filename.stem
    # Extract the part after the last underscore
    return basename.split("_")[-1]

# Iterate over all pairs of files (tuples)
for i in range(len(files)):
    for j in range(i + 1, len(files)):
        file1 = files[i]
        file2 = files[j]

        # Extract model names
        model1 = extract_model_name(file1)
        model2 = extract_model_name(file2)

        if compare_to_gt_only and (model1 != "gt" and model2 != "gt"):
            continue
        
        # Print the tuple of model names
        print(f"Compare the two models: ({model1}, {model2})")
        output_file_path = output_directory / f"ir_{model1}_{model2}.json"
        
        if model1 == model2:
            print("Models are identical. Skipping.")
            continue
        
        compute_interrater_reliability_metrics(
            path_to_files=([file1, file2]),
            output_file_path=output_file_path,
            aggregation=aggregation,
        )
        print(f"Metrics successfully written to {output_file_path}")
