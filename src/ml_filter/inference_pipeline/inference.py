import argparse
import json
import logging
import os

import torch
from optimum.bettertransformer import BetterTransformer
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Distributed inference with BERT")
    parser.add_argument('--input_files_list', type=str,
                        help='Path to the text file containing list of input JSONL files')
    parser.add_argument('--output_file', type=str, help='Output file for results')
    parser.add_argument('--checkpoint_file', type=str, help='Checkpoint file to save progress', default=None)
    parser.add_argument('--task_id', type=int, default=0, help='Task ID (from 0 to N-1)')
    parser.add_argument('--num_tasks', type=int, default=1, help='Total number of tasks')
    parser.add_argument('--batch_size', type=int, default=512, help='Batch size for inference')
    args = parser.parse_args()
    return args


def setup_logging(task_id):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(f'logs/task_{task_id}.log')
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - Task %(task_id)d - %(levelname)s - %(message)s', style='%')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_checkpoint(checkpoint_file):
    if checkpoint_file and os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
        return checkpoint
    else:
        return {}


def save_checkpoint(checkpoint_file, checkpoint_data):
    temp_file = checkpoint_file + '.tmp'
    with open(temp_file, 'w') as f:
        json.dump(checkpoint_data, f)
    os.replace(temp_file, checkpoint_file)


def main():
    args = parse_args()

    logger = setup_logging(args.task_id)
    logger.info(f"Starting task {args.task_id} out of {args.num_tasks} tasks.")

    # Get the assigned GPU from CUDA_VISIBLE_DEVICES
    device = 0 if torch.cuda.is_available() else -1

    logger.info("Loading model and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
    model = AutoModelForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=5,
        torch_dtype=torch.bfloat16
    )
    model = BetterTransformer.transform(model)
    classifier = pipeline(
        'text-classification',
        model=model,
        tokenizer=tokenizer,
        device=device,
        batch_size=args.batch_size
    )

    logger.info("Model loaded successfully.")

    with open(args.input_files_list, 'r') as f:
        input_files = [line.strip() for line in f if line.strip()]

    logger.info(f"Total input files: {len(input_files)}")

    checkpoint_file = args.checkpoint_file or f'checkpoints/checkpoint_task{args.task_id}.json'
    checkpoint = load_checkpoint(checkpoint_file)
    logger.info(f"Loaded checkpoint: {checkpoint}")

    with open(args.output_file, 'a') as outfile:
        batch_texts = []
        batch_items = []
        total_lines = 0
        processed_lines = checkpoint.get('processed_lines', 0)
        current_file_index = checkpoint.get('current_file_index', 0)
        current_line_index = checkpoint.get('current_line_index', 0)

        # Process data starting from the checkpoint
        for file_idx, input_file in enumerate(input_files[current_file_index:], start=current_file_index):
            logger.info(f"Processing file: {input_file}")
            with open(input_file, 'r') as infile:
                for idx, line in enumerate(infile):
                    total_lines += 1
                    # Skip lines until we reach the checkpointed line
                    if file_idx == current_file_index and idx < current_line_index:
                        continue

                    # Assign lines to tasks based on line index modulo number of tasks
                    if idx % args.num_tasks != args.task_id:
                        continue  # Skip lines not assigned to this task

                    item = json.loads(line)
                    text = item['text']

                    batch_texts.append(text)
                    batch_items.append(item)

                    # When batch is full, process it
                    if len(batch_texts) == args.batch_size:
                        # Perform inference on batch
                        results = classifier(batch_texts)
                        # Write results
                        for item, result in zip(batch_items, results):
                            score_label = result['label'].split('_')[-1]
                            output_item = {
                                'text': item['text'],
                                'score': int(score_label),
                                'confidence': result['score']
                            }
                            outfile.write(json.dumps(output_item) + '\n')
                            processed_lines += 1

                        batch_texts = []
                        batch_items = []
                        logger.info(f"Processed {processed_lines} lines.")

                        checkpoint_data = {
                            'processed_lines': processed_lines,
                            'current_file_index': file_idx,
                            'current_line_index': idx + 1
                        }
                        save_checkpoint(checkpoint_file, checkpoint_data)
                        logger.info(f"Checkpoint saved: {checkpoint_data}")

                # Process any remaining items in the batch after file ends
                if batch_texts:
                    results = classifier(batch_texts)
                    for item, result in zip(batch_items, results):
                        score_label = result['label'].split('_')[-1]
                        output_item = {
                            'text': item['text'],
                            'score': int(score_label),
                            'confidence': result['score']
                        }
                        outfile.write(json.dumps(output_item) + '\n')
                        processed_lines += 1
                    batch_texts = []
                    batch_items = []
                    logger.info(f"Processed {processed_lines} lines.")

                    checkpoint_data = {
                        'processed_lines': processed_lines,
                        'current_file_index': file_idx + 1,
                        'current_line_index': 0
                    }
                    save_checkpoint(checkpoint_file, checkpoint_data)
                    logger.info(f"Checkpoint saved: {checkpoint_data}")

    logger.info(f"Task {args.task_id} completed. Total lines processed: {processed_lines}")


if __name__ == '__main__':
    main()
