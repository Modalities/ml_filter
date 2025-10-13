# Embedding & Annotation Pipelines

This document explains how to generate model embeddings for large JSONL corpora and then run regression / classification heads to obtain annotation scores at scale using MLFilter's Datatrove-based pipelines.

## Installation

Install ML-Filter (editable for development) and optionally with CUDA 12.6 wheels for PyTorch.

Basic editable install (CPU or existing CUDA env):
```bash
pip install -e .
```

Install with CUDA 12.6 extra (will pull matching PyTorch wheels):
```bash
pip install .[cuda126] --extra-index-url https://download.pytorch.org/whl/cu126
```

Notes:
- The `cuda126` extra expects a GPU environment with CUDA 12.6 capable drivers.
- If you already have a suitable PyTorch installed, you can omit the extra and just use the editable install.
- For development you may also want to install any project specific dev/test extras if they exist (see `pyproject.toml`).

## Overview

The workflow consists of two sequential pipelines:

1. Embedding Pipeline (`run_embedding_pipeline`)  
   Reads raw JSONL documents, tokenizes & feeds them through an embedding model, and stores embeddings (optionally with labels) into per-source HDF5 files.
2. Annotation Pipeline (`run_annotation_pipeline`)  
   Reads the produced HDF5 embedding files and applies one or more trained regression/classification heads to generate new annotated JSONL outputs.

Each pipeline can run locally (single or multi-process) or on a Slurm cluster (array jobs) using unified YAML configuration schemas validated by Pydantic models.

## Directory Conventions

```
<project_root>/
  data/
  outputs/
    embeddings/          # embedding_output_dir (per YAML)
      file1.h5
      file2.h5
    annotated_data/      # annotation outputs (per YAML)
      file1.jsonl
      file2.jsonl
  configs/
    embedding_job.yaml
    annotation_job.yaml
```

You decide names/paths via YAML `params` sections (see below).

---
## Embedding Pipeline

### YAML Schema (`EmbeddingPipelineParameters`)
Key fields under the top-level `params:` section:

| Field | Type | Description |
|-------|------|-------------|
| `input_dir` | str | Directory containing source JSONL files. |
| `glob_pattern` | str | Glob selecting which JSONL files to process (e.g. `*.jsonl`). |
| `keys_to_index` | list[str] | JSON keys to extract & concatenate / embed (e.g. `["text"]`). |
| `output_dir` | path | Base output directory. |
| `embedding_dir` | str | Subdirectory (relative to `output_dir`) for HDF5 files. |
| `embedding_model` | str | HF model id or local path for embedding model. |
| `hdf5_dataset_name` | str | Name of dataset group inside HDF5 (default `train`). |
| `batch_size` | int | Batch size for forward embedding model passes. |
| `writer_batch_size` | int | Accumulation size before flushing to disk. |
| `max_length` | int | Max token length for truncation. |
| `padding` | bool/str | Padding strategy passed to tokenizer. |
| `truncation` | bool/str | Truncation strategy. |
| `save_labels` | bool | If true, propagate existing labels from JSONL into HDF5 metadata. |

### Execution Settings
Top-level booleans & sections:
- `running_on_slurm`: bool (defaults to false if absent).  
- `local_settings`: only when `running_on_slurm: false`.
- `slurm_settings`: only when `running_on_slurm: true`.

Minimal local example:
```yaml
running_on_slurm: false
params:
  input_dir: data/jsonl
  glob_pattern: "*.jsonl"
  keys_to_index: ["text"]
  output_dir: outputs
  embedding_dir: embeddings
  embedding_model: sentence-transformers/all-MiniLM-L6-v2
  hdf5_dataset_name: train
  batch_size: 64
  writer_batch_size: 512
  max_length: 512
  padding: true
  truncation: true
  save_labels: true
local_settings: {}
```

Slurm example (excerpt):
```yaml
running_on_slurm: true
params:
  input_dir: /scratch/corpus
  glob_pattern: "*.jsonl"
  keys_to_index: ["title", "body"]
  output_dir: /scratch/run123
  embedding_dir: embeddings
  embedding_model: my-org/embedding-large
  hdf5_dataset_name: train
  batch_size: 128
  writer_batch_size: 1024
  max_length: 1024
  padding: longest
  truncation: true
  save_labels: false
slurm_settings:
  tasks: 200        # number of shards / array jobs
  time: "08:00:00"
  partition: gpu
  cpus_per_task: 8
  mem_per_cpu_gb: 8
  job_name: emb_job
  sbatch_args: { gres: "gpu:1" }
```

### Running
```bash
ml_filter run_embedding_pipeline --config_file_path configs/embedding_job.yaml
```
(Or invoke `run_embedding_pipeline(Path("..."))` programmatically.)

### Outputs
One HDF5 per input JSONL file with structure:
- dataset name = `hdf5_dataset_name`
- columns: `embeddings`, optional `labels`, index metadata (original keys).

---
## Annotation Pipeline

Consumes embeddings and applies trained Pytorch regression/classification heads.

### YAML Schema (`AnnotationPipelineParameters`)

| Field | Type | Description |
|-------|------|-------------|
| `embeddings_directory` | str | Directory containing produced HDF5 embedding files. |
| `output_keys` | list[str] | Keys / metadata fields to write into output JSONL. |
| `output_dir` | path | Base output directory (annotated data → `output_dir/annotated_data`). |
| `regression_head_checkpoints` | dict[str,str] | Map model name → path to head checkpoint (can specify multiple heads). |
| `batch_size` | int | Batch size for head forward passes. |

Execution mode fields mirror embedding pipeline: `running_on_slurm`, `local_settings` or `slurm_settings`.

### Minimal Local Example
```yaml
running_on_slurm: false
params:
  embeddings_directory: outputs/embeddings
  output_keys: ["doc_id", "text"]
  output_dir: outputs
  regression_head_checkpoints:
    quality: checkpoints/quality_head.pt
    toxicity: checkpoints/tox_head.pt
  batch_size: 256
local_settings:
  tasks: 1
  local_tasks: 1
  local_rank_offset: 0
  workers: 4
```

### Running
```bash
ml_filter run_annotation_pipeline --config_file_path configs/annotation_job.yaml
```

### Outputs
Per embedding source file: `${source_filename}.jsonl` written to:
```
<output_dir>/annotated_data/
```
Each line contains original metadata (from `output_keys`) plus head outputs (scores / predictions).

---
## Chaining the Pipelines

1. Generate embeddings: `run_embedding_pipeline` → HDF5 files.  
2. Train or supply head checkpoints externally.  
3. Run `run_annotation_pipeline` referencing the embedding directory and head checkpoint paths.  
4. Downstream: aggregate / evaluate via existing CLI commands (`aggregate_scores`, `evaluate_predicted_annotations`).

---
## Slurm Notes
- `tasks` controls sharding; each task processes a partition of input files.  
- Provide GPU-related sbatch args via `sbatch_args` (e.g., `{ gres: "gpu:1" }`).  
- Logging directories auto-default to `<output_dir>/logs` if not set.  
- For very large corpora tune: `batch_size`, `writer_batch_size`, and number of `tasks`.

---
## Common Pitfalls & Tips
- Ensure `keys_to_index` actually exist in your JSONL lines; missing keys may raise errors.  
- Use consistent tokenizer / preprocessing across embedding and head training.  
- If adding more files later, you can re-run embedding pipeline; already existing HDF5s can be guarded by a future `skip_completed` enhancement (present for Slurm runs in embedding pipeline via executor).  
- Keep head checkpoints versioned; mismatches between embedding model and head can silently reduce quality.  
- Prefer smaller `writer_batch_size` if you see high memory usage.  
- When running on Slurm with large `tasks`, watch scheduler array limits (`max_array_size`).

---
## Programmatic Usage Sketch
```python
from pathlib import Path
from ml_filter.annotation.embedding_pipeline import run_embedding_pipeline
from ml_filter.annotation.annotation_pipeline import run_annotation_pipeline

run_embedding_pipeline(Path("configs/embedding_job.yaml"))
run_annotation_pipeline(Path("configs/annotation_job.yaml"))
```
