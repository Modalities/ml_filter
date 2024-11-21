import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def process_dump(
    data_folder: str,
    model_name: str,
    output_path: str,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    lengths = []
    n_errors = 0
    root_directory = os.fsencode(data_folder)
    file_count = sum(len(files) for _, _, files in os.walk(root_directory))
    with tqdm(total=file_count) as pbar:
        for root, dirs, files in os.walk(root_directory):
            for filename in files:
                pbar.update(1)
                filename = os.path.join(root, filename)
                if filename.endswith(b".jsonl"):
                    # print(f"processing {filename}")
                    with open(filename, "r", encoding="utf-8", errors="ignore") as f:
                        for line in f:
                            if line.strip():
                                try:
                                    text = json.loads(line)["text"]
                                    lengths.append(len(tokenizer(text)["input_ids"]))
                                except json.decoder.JSONDecodeError as e:
                                    print(f"Error reading {filename}: {e}")
                                    n_errors += 1
                else:
                    continue
    with open(output_path, "w") as f:
        json.dump(lengths, f, indent=2)
    print(f"Encountered {n_errors} JSON formatting errors in {len(lengths)} files")


def compute_stats(lengths_path: str):
    with open(lengths_path, "r") as f:
        lengths = json.load(f)
    print(f"Mean: {np.mean(lengths)}, std: {np.std(lengths)}, median: {np.median(lengths)}")
    print(f"Fraction shorter than 512: {(np.array(lengths) < 512).mean()}")


def plot_histogram(lengths_path: str, bins: int, save_path: str):
    with open(lengths_path, "r") as f:
        lengths = json.load(f)
    counts, bins, _ = plt.hist(lengths, bins=bins)
    plt.axvline(np.median(lengths), color="k", linestyle="--", label="median")
    plt.axvline(512, color="r", linestyle="--", label="context length")
    plt.axvline(np.percentile(lengths, 95), color="k", linestyle=":", label="95th percentile of length")
    plt.xlabel("Document length (visible text tokens)")
    plt.ylabel("Count")
    plt.title("Document length distribution for {lengths_path}}")
    plt.xlim(-10, 5000)
    plt.legend()
    plt.savefig(save_path)


def main():
    output_path = "./data/cc_subset_10000_docs.json"
    process_dump(
        data_folder="/raid/s3/opengptx/max_lue/eurolingua/data/cc_subset_10000_docs",
        model_name="FacebookAI/xlm-roberta-large",
        output_path=output_path,
    )
    compute_stats(output_path)
    plot_histogram(output_path, 2000, "./data/cc_subset_10000_docs.png")


if __name__ == "__main__":
    main()
