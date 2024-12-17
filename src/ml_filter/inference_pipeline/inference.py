import argparse
import json
import logging
import os
from typing import List, Dict, Set

import torch
from datasets import load_from_disk, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Parallel Inference with BERT")
    parser.add_argument('--input_files_list', type=str, help='Path to the list of input dataset files')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Directory for output files')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='Checkpoint directory')
    parser.add_argument('--task_id', type=int, default=0, help='Task ID for parallel execution')
    parser.add_argument('--num_tasks', type=int, default=1, help='Total number of tasks')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for inference')
    return parser.parse_args()


def setup_logging(task_id: int) -> logging.Logger:
    log_dir: str = "logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(log_dir, f"task_{task_id}.log"),
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(f"task_{task_id}")


def collate_fn(batch: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
    input_ids = [torch.tensor(example['input_ids'], dtype=torch.long) for example in batch]
    attention_mask = [torch.tensor(example['attention_mask'], dtype=torch.long) for example in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = pad_sequence(attention_mask, batch_first=True, padding_value=0)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    logger: logging.Logger = setup_logging(args.task_id)
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("Loading model...")
    model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)
    model.to(device).eval()

    with open(args.input_files_list, 'r') as f:
        input_files: List[str] = [line.strip() for line in f]

    checkpoint_file: str = os.path.join(args.checkpoint_dir, f"checkpoint_task{args.task_id}.json")
    completed_files: Set[str] = set()

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            completed_files = set(json.load(f))

    for file_path in input_files:
        if file_path in completed_files:
            logger.info(f"Skipping already processed file: {file_path}")
            continue

        logger.info(f"Processing file: {file_path}")
        dataset: Dataset = load_from_disk(file_path).shard(args.num_tasks, args.task_id)
        dataloader: DataLoader = DataLoader(dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        output_file: str = os.path.join(args.output_dir, f"output_task{args.task_id}.jsonl")
        with torch.no_grad(), open(output_file, 'a') as f:
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                predictions = torch.argmax(outputs.logits, dim=-1)

                for pred in predictions.cpu().tolist():
                    f.write(json.dumps({"prediction": pred}) + '\n')

        completed_files.add(file_path)
        with open(checkpoint_file, 'w') as f:
            json.dump(list(completed_files), f)

    logger.info(f"Task {args.task_id} completed successfully.")


if __name__ == '__main__':
    main()
