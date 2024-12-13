
import json
import os
import random
from typing import List
from datasets import load_dataset
from huggingface_hub import HfApi


HF_TOKEN = os.environ["HF_TOKEN"]


def sample_from_hf_dataset(
    dataset_name: str,
    dataset_split: str,
    output_file_path: str,
    hf_repo_path: str,
    hf_repo_id: str,
    column_name: str,
    relevant_classes: List[int],
    num_docs_per_class: int,
    seed: int
):
    """
    Samples documents from a Hugging Face dataset based on specified classes and uploads the results to Hugging Face Hub.

    Args:
        dataset_name (str): The name of the Hugging Face dataset to sample from.
        dataset_split (str): The split of the Hugging Face dataset that is used for sampling.
        output_file_path (str): Path to save the sampled data as a JSON file.
        hf_repo_path (str): The path in the Hugging Face Hub repository where the file will be stored.
        hf_repo_id (str): The ID of the Hugging Face repository to upload the file to.
        column_name (str): The column in the dataset used for filtering (e.g., "score").
        relevant_classes (List[int]): List of class values to filter and sample from.
        num_docs_per_class (int): Number of documents to sample for each class.
        seed (int): Seed value for random operations to ensure reproducibility.

    Returns:
        None
    """
    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(dataset_name)

    # Processing: Sample documents for each relevant score
    print(f"Sampling {num_docs_per_class} documents per unique value in column '{column_name}'...")
    sampled_data = []

    for value in relevant_classes:
        group = dataset[dataset_split].filter(lambda x: x[column_name] == value)
        # Randomly sample from the group (handle cases where group size < SAMPLES_PER_GROUP)
        sampled_group = group.shuffle(seed=seed).select(range(min(len(group), num_docs_per_class)))
        sampled_data.extend(sampled_group)

    # shuffle data
    random.seed(seed)
    random.shuffle(sampled_data)


    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in sampled_data:
            json.dump(item, f, ensure_ascii=False)
            f.write("\n")

    # upload json file to huggingface
    api = HfApi()
    api.upload_file(
        path_or_fileobj=output_file_path,
        path_in_repo=hf_repo_path,
        repo_id=hf_repo_id,
        repo_type="dataset",
        token=HF_TOKEN,
    )
