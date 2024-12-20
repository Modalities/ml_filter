import json
import logging
import random
from typing import Dict, List
from datasets import load_dataset
from huggingface_hub import HfApi


def sample_from_hf_dataset(
    dataset_name: str,
    dataset_split: str,
    output_file_path: str,
    column_name: str,
    column_values: List[str],
    num_docs_per_value: int,
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
        column_values (List[str]): List of column values to filter and sample from as strings.
        num_docs_per_value (int): Number of documents to sample for each column value.
        seed (int): Seed value for random operations to ensure reproducibility.

    Returns:
        None
    """
    logging.basicConfig(level=logging.INFO)
    
    # Load dataset
    logging.info("Loading dataset...")
    dataset = load_dataset(dataset_name)

    # Processing: Sample documents for each relevant score
    logging.info(f"Sampling {num_docs_per_value} documents per unique value in column '{column_name}'...")
    sampled_data = []

    for value in column_values:
        group = dataset[dataset_split].filter(lambda x: str(x[column_name]) == value)
        # Randomly sample from the group (handle cases where group size < SAMPLES_PER_GROUP)
        sampled_group = group.shuffle(seed=seed).select(range(min(len(group), num_docs_per_value)))
        sampled_data.extend(sampled_group)

    # shuffle data
    random.seed(seed)
    random.shuffle(sampled_data)

    # write sampled data to jsonl file
    save_data_to_file(
        output_file_path=output_file_path,
        data=sampled_data,
        encoding="utf-8",
        ensure_ascii=False
        )


def upload_file_to_hf(file_path: str, hf_repo_path: str, hf_repo_id: str, repo_type: str, hf_token: str):
    """
    Uploads a file to the Hugging Face Hub.

    Args:
        file_path (str): The local path to the file to be uploaded.
        hf_repo_path (str): The path in the Hugging Face repository where the file will be stored.
        hf_repo_id (str): The ID of the Hugging Face repository.
        repo_type (str): The type of the repository (default is "dataset").
        hf_token (str): The Hugging Face authentication token (default is taken from the environment variable "HF_TOKEN").

    Returns:
        None
    """
    api = HfApi()
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=hf_repo_path,
        repo_id=hf_repo_id,
        repo_type=repo_type,
        token=hf_token,
    )
    
    
def save_data_to_file(output_file_path: str, data: list[dict], encoding: str="utf-8", ensure_ascii: bool = False):
    with open(output_file_path, "w", encoding=encoding) as f:
        for item in data:
            json.dump(item, f, ensure_ascii=ensure_ascii)
            f.write("\n")
