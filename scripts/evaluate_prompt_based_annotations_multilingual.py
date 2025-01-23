from pathlib import Path
from ml_filter.analysis.interrater_reliability import compute_interrater_reliability_metrics


#### Parameters ####
input_directory = Path("/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/experiments/multilinguality/experiments")
output_directory = input_directory / "comparison"
gt_data = Path('/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/experiments/model_size_architecture/annotations/annotations_edu_en_gt.jsonl')
aggregation = "majority"
####################1

# Find all files matching the pattern in the directory and subdirectories
files = list(input_directory.rglob("annotations_*.jsonl"))

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
for file in files:
    # Extract model names
    model = extract_model_name(file)
    lang = file.parent.name
    
    # Print the tuple of model names
    print(f"Compare model {model} to ground truth")
    lang_dir = output_directory / lang
    lang_dir.mkdir(parents=True, exist_ok=True)
    
    compute_interrater_reliability_metrics(
        path_to_files=([gt_data, file]),
        output_dir=lang_dir,
        aggregation=aggregation,
        gt_file_idx=0,
        model_name=model,
    )
    print(f"Metrics successfully written to {lang_dir}")
