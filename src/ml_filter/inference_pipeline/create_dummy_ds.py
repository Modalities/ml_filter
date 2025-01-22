import os

import numpy as np
from datasets import Dataset


def create_dummy_dataset(num_examples: int, seq_length: int, save_path: str | os.PathLike):
    data = {
        'input_ids': [np.random.randint(0, 30522, seq_length).tolist() for _ in range(num_examples)],
        'attention_mask': [[1] * seq_length for _ in range(num_examples)],
    }
    dataset = Dataset.from_dict(data)
    dataset.save_to_disk(save_path)


# Example usage:
# Create the directory in the root folder - going up 3 levels from current directory
root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
save_dir = os.path.join(root_dir, 'data', 'inference_pipeline')
os.makedirs(save_dir, exist_ok=True)
create_dummy_dataset(num_examples=10000, seq_length=512, save_path=os.path.join(save_dir, 'dummy_dataset1'))
