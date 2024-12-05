
import json
import os
import random
from datasets import load_dataset
from huggingface_hub import HfApi


# Configuration
DATASET_NAME = "HuggingFaceFW/fineweb-edu-llama3-annotations"
OUTPUT_PATH = "fineweb_gt_test_no_ascii_utf8.json"
HF_REPO_PATH = "fineweb_gt_test_no_ascii_utf8.json"
COLUMN_NAME = "score"
NUM_DOCS_PER_SCORE = 2
RELEVANT_SCORES = [0, 1, 2, 3, 4]
HF_TOKEN = os.environ["HF_TOKEN"]
SEED = 42

# Load dataset
print("Loading dataset...")
dataset = load_dataset(DATASET_NAME)

# Processing: Sample documents for each relevant score
print(f"Sampling {NUM_DOCS_PER_SCORE} documents per unique value in column '{COLUMN_NAME}'...")
sampled_data = []

for value in RELEVANT_SCORES:
    group = dataset["train"].filter(lambda x: x[COLUMN_NAME] == value)
    # Randomly sample from the group (handle cases where group size < SAMPLES_PER_GROUP)
    sampled_group = group.shuffle(seed=SEED).select(range(min(len(group), NUM_DOCS_PER_SCORE)))
    sampled_data.extend(sampled_group)

# shuffle data
random.seed(SEED)
random.shuffle(sampled_data)

# convert data to dict and save it to json file
sampled_data_dict = {}
for k in ["text", "metadata", "prompt", "score"]:
    sampled_data_dict[k] = {str(i): v[k] for i, v in enumerate(sampled_data)}

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(sampled_data_dict, f, ensure_ascii=False)

# upload json file to huggingface
api = HfApi()
api.upload_file(
    path_or_fileobj=OUTPUT_PATH,
    path_in_repo=HF_REPO_PATH,
    repo_id="Eurolingua/ml_filter",
    repo_type="dataset",
    token=HF_TOKEN,
)
