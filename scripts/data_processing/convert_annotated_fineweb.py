import json
from datasets import load_dataset
import os
from pathlib import Path
import random, math

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

def multi_score_transform(transform_fns):
    """Transform single scores into multiple scores using different transformations.
    
    Args:
        transform_fns: List of tuples containing (name, transform_function) pairs.
                      The original score will always be kept.
    """
    if not transform_fns:
        raise ValueError("At least one transform function must be provided")
    
    # Always include original score first
    transforms = [("score", lambda x: x)] + transform_fns
    
    input_file = Path("data/annotated_fineweb.jsonl")
    output_file = Path("data/annotated_fineweb_multi.jsonl")
    
    print("Applying score transformations...")
    with open(input_file, 'r', encoding='utf-8') as in_f, \
         open(output_file, 'w', encoding='utf-8') as out_f:
        
        for line in in_f:
            entry = json.loads(line)
            original_score = entry['score']
            
            # Apply all transformations
            entry['scores'] = {
                name: transform_fn(original_score) 
                for name, transform_fn in transforms
            }
            
            # Remove the original single score field
            del entry['score']
            
            # Write transformed entry
            out_f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"Score transformation complete! File saved to {output_file}")

def split_dataset(file_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """Split the dataset into train, validation and test sets."""
    input_file = Path(file_path + ".jsonl")
    
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
    train_file = Path(f"{file_path}_train.jsonl")
    val_file = Path(f"{file_path}_val.jsonl")
    test_file = Path(f"{file_path}_test.jsonl")
    
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
    #   convert_to_jsonl()
    multi_score_transform(transform_fns=[
        ("score_transform_1", lambda x: min(x * 2, 5)),  # Double score capped at 5
        ("score_transform_2", lambda x: min(max(x * 1.5 + random.uniform(-0.2, 0.2) + math.sin(x/2), 0), 5)),  # Complex with randomness
        ("score_transform_3", lambda x: 1 if math.sin(x*math.pi/2.5) + math.cos(x) + random.uniform(-0.1, 0.1) > 0.5 else 0)  # Complex binary transform
    ])
    split_dataset("data/annotated_fineweb_multi")