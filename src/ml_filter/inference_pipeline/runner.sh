#!/bin/bash
#SBATCH --job-name=bert_inference
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/inference_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bert-env

mkdir -p logs checkpoints outputs

INPUT_FILES_LIST="input_files_list.txt"

srun python inference.py \
  --input_files_list $INPUT_FILES_LIST \
  --output_dir outputs \
  --checkpoint_dir checkpoints \
  --task_id $SLURM_PROCID \
  --num_tasks $SLURM_NTASKS
