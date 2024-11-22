# Distributed BERT Inference with Checkpointing

This repository provides code to perform distributed inference using BERT on a Slurm cluster with multiple GPUs and
nodes. It includes optimizations for faster inference and checkpointing to allow resuming from interruptions.

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/EuroLingua-GPT/ml_filter
cd ml_filter/src/ml_filter/inference_pipeline
```

### 2. Create and Activate a Conda Environment

```bash
conda create -n bert-env python=3.10.0
conda activate bert-env
```

### 3. Install Required Python Packages

```
conda install pytorch torchvision torchaudio cudatoolkit=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 4. Prepare Input Files List

Create a text file named input_files.txt containing the paths to your input JSONL files, one per line.
The jsonl files assumes the data is at the "text" key.

```bash
/data/jsonl/file1.jsonl
/data/jsonl/file2.jsonl
/data/jsonl/file3.jsonl
...
```

## Usage

### 1. Launch job

```bash
conda activate bert-env
chmod +x run_inference.slurm
sbatch run_inference.slurm
```

### 2. Collect results

```bash
cat outputs/output_task*.jsonl > combined_output.jsonl
```