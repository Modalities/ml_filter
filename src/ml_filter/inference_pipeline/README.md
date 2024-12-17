# BERT Inference Pipeline

This repository runs large-scale parallel inference on tokenized datasets using BERT.

## 1. Setup

### Install Dependencies
```bash
conda create -n bert-env python=3.11
conda activate bert-env
pip install -r requirements.txt
```

## 2. Create Dummy Datasets
Use `create_dummy_ds.py` to generate tokenized datasets:
```bash
python create_dummy_ds.py
```

## 3. Running the Pipeline
### Local Execution
##### Run inference on a single machine:

```bash
python inference.py --input_files_list input_files_list.txt --output_dir outputs --checkpoint_dir checkpoints --task_id 0 --num_tasks 1
```
### SLURM Execution
##### Submit the job to a SLURM cluster:
```bash
sbatch runner.sh
```
## 4. Output
```bash
Logs: logs/
Outputs: outputs/
Checkpoints: checkpoints/
```
## 5. Run tests
To validate key components of the pipeline (e.g., collate_fn and dataset sharding), run:
```bash
python -m unittest test_inference.py
```

## Guesstimate of inference time for 93 CC dumps
```
| Model                | Single GPU   | 64 GPUs (16 nodes x 4 GPUs) |
|----------------------|--------------|-----------------------------|
| BERT Base            | 430 days     | ~6.7 days                   |
| DistilBERT           | 215 days     | ~3.4 days                   |
| XLM-RoBERTa Large    | 2,150 days   | ~33.6 days                  |
```
