# This script converts the HuggingFaceFW/fineweb-edu-llama3-annotations dataset into a JSONL format that is compatible with the ML Filter Classifier.
# It also creates a multi-score version of the dataset by applying various transformations to the original score.

import json
from datasets import load_dataset
import os
from pathlib import Path
import random, math
from typing import List, Tuple, Callable, Union
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def convert_to_jsonl(base_path: str):
    # Load the dataset
    logger.info("Loading dataset...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu-llama3-annotations")
    
    # Create output directory if it doesn't exist
    os.makedirs(base_path, exist_ok=True)
    
    # Open output file
    logger.info("Converting to JSONL format...")
    output_file = os.path.join(base_path, "annotated_fineweb.jsonl")
    with open(output_file, "w", encoding="utf-8") as f:
        # Process each example
        for idx, example in enumerate(dataset['train']): # note: this dataset only has a "train" split
            # Create entry in desired format
            entry = {
                "id": str(idx),
                "text": example['text'],
                "scores": {
                    "score": example['score']
                }
            }
            
            # Write to file
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    logger.info(f"Conversion complete! File saved to {output_file}")
    
    # Delete downloaded dataset cache
    logger.info("Cleaning up downloaded data...")
    dataset.cleanup_cache_files()
    logger.info("Done!")

def multi_score_transform(base_path: str, transform_fns: List[Tuple[str, Callable[[float], Union[int, float]]]]):
    """Transform single scores into multiple scores using different transformations.
    
    Args:
        base_path: Base directory path for data files
        transform_fns: List of tuples containing (name, transform_function) pairs.
                      The original score will always be kept.
    """
    
    # Always include original score first
    transforms = [("score", lambda x: int(round(x)))] + transform_fns
    
    input_file = Path(os.path.join(base_path, "annotated_fineweb.jsonl"))
    output_file = Path(os.path.join(base_path, "annotated_fineweb_multi.jsonl"))
    
    logger.info("Applying score transformations...")
    with open(input_file, 'r', encoding='utf-8') as in_f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        for line in in_f:
            entry = json.loads(line)
            original_score = entry['scores']['score']
            
            # Apply all transformations
            entry['scores'] = {
                name: transform_fn(original_score) 
                for name, transform_fn in transforms
            }
            
            # Write transformed entry
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    logger.info(f"Score transformation complete! File saved to {output_file}")

def split_dataset(base_path: str, file_path: str, train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42):
    """Split the dataset into train, validation and test sets."""
    input_file = Path(os.path.join(base_path, file_path + ".jsonl"))
    
    # Count total number of samples
    with open(input_file, 'r', encoding='utf-8') as f:
        num_samples = sum(1 for _ in f)
    
    # Calculate split sizes
    train_size = int(num_samples * train_ratio)
    val_size = int(num_samples * val_ratio)
    test_size = num_samples - train_size - val_size
    
    # Generate shuffled indices
    random.seed(seed)
    indices = list(range(num_samples))
    random.shuffle(indices)
    
    # Split indices
    train_indices = set(indices[:train_size])
    val_indices = set(indices[train_size:train_size + val_size])
    test_indices = set(indices[train_size + val_size:])
    
    # Create output files
    train_file = Path(os.path.join(base_path, f"{file_path}_train.jsonl"))
    val_file = Path(os.path.join(base_path, f"{file_path}_val.jsonl"))
    test_file = Path(os.path.join(base_path, f"{file_path}_test.jsonl"))
    
    # Open output files
    with open(train_file, 'w', encoding='utf-8') as train_f, \
         open(val_file, 'w', encoding='utf-8') as val_f, \
         open(test_file, 'w', encoding='utf-8') as test_f, \
         open(input_file, 'r', encoding='utf-8') as in_f:
        
        for idx, line in enumerate(in_f):
            if idx in train_indices:
                train_f.write(line)
            elif idx in val_indices:
                val_f.write(line)
            else:
                test_f.write(line)
    
    logger.info(f"Split complete! Created files:")
    logger.info(f"Train ({train_size} samples): {train_file}")
    logger.info(f"Validation ({val_size} samples): {val_file}") 
    logger.info(f"Test ({test_size} samples): {test_file}")

if __name__ == "__main__":
    base_path = "data"  # Can be changed to any desired path

    # download data and create single score file
    convert_to_jsonl(base_path)
    split_dataset(base_path, "annotated_fineweb")

    # create multi-score file
    multi_score_transform(base_path, transform_fns=[
        ("score_transform_1", lambda x: min(x + 1, 5)),  # shift up by 1, cap at 5
        ("score_transform_2", lambda x: min(max(x + random.uniform(-0.5, 0.5), 0), 5)),  # add random noise between -0.5 and 0.5, clamp to [0,5]
        ("score_transform_3", lambda x: 1 if x >= 3 else 0)  # binary threshold at 3
    ])
    split_dataset(base_path, "annotated_fineweb_multi")