# BERT Inference Pipeline
This repository runs large-scale parallel inference on tokenized datasets using sequence classification models, such as Roberta.

## 1. Setup

### Install Dependencies
This code is compatible with the requirements of the ML Filter project. Please see [../../../CONTRIBUTING.md](../../../CONTRIBUTING.md) for more information on the installation.

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
## Guesstimate of inference time for 93 CC dumps
```
| Model                | Single GPU   | 64 GPUs (16 nodes x 4 GPUs) |
|----------------------|--------------|-----------------------------|
| BERT Base            | 430 days     | ~6.7 days                   |
| DistilBERT           | 215 days     | ~3.4 days                   |
| XLM-RoBERTa Large    | 2,150 days   | ~33.6 days                  |
```
