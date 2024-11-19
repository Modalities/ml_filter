import json
from datasets import load_dataset
import os
from pathlib import Path
import random

def convert_to_jsonl():
    # Load the dataset
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceFW/fineweb-edu-llama3-annotations")
    
    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)
    
    # Open output file
    print("Converting to JSONL format...")
    with open("data/annotated_fineweb.jsonl", "w", encoding="utf-8") as f:
        # Process each example
        for idx, example in enumerate(dataset['train']): # note: this dataset only has a "train" split
            # Create entry in desired format
            entry = {
                "id": str(idx),
                "text": example['text'],
                "score": example['score']
            }
            
            # Write to file
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Conversion complete! File saved to data/annotated_fineweb.jsonl")
    
    # Delete downloaded dataset cache
    print("Cleaning up downloaded data...")
    dataset.cleanup_cache_files()
    print("Done!")

def split_dataset(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split the dataset into train, validation and test sets."""
    input_file = Path("data/annotated_fineweb.jsonl")
    
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
    train_file = Path("data/annotated_fineweb_train.jsonl")
    val_file = Path("data/annotated_fineweb_val.jsonl")
    test_file = Path("data/annotated_fineweb_test.jsonl")
    
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
    
    print(f"Split complete! Created files:")
    print(f"Train ({train_size} samples): {train_file}")
    print(f"Validation ({val_size} samples): {val_file}") 
    print(f"Test ({test_size} samples): {test_file}")

if __name__ == "__main__":
    convert_to_jsonl()
    split_dataset()
