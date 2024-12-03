#!/bin/bash
#SBATCH --job-name=bert_inference
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/inference_%j_%t.out
#SBATCH --error=logs/inference_%j_%t.err

#module load python/3.10
module load cuda/12.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate bert-env

mkdir -p logs
mkdir -p outputs
mkdir -p checkpoints

NUM_TASKS=$SLURM_NTASKS

# Path to a txt file with a list of jsonl paths
INPUT_FILES_LIST="input_files.txt"

# Checkpoints of which lines are processed, so we dont start over
CHECKPOINT_FILE="checkpoints/checkpoint_task${SLURM_PROCID}.json"

# Run the Python script using srun
srun --ntasks=$NUM_TASKS --ntasks-per-node=4 --cpus-per-task=4 \
  --gpus-per-task=1 --exclusive --cpu-bind=none \
  python inference.py --input_files_list $INPUT_FILES_LIST \
  --output_file outputs/output_task${SLURM_PROCID}.jsonl \
  --checkpoint_file $CHECKPOINT_FILE \
  --task_id $SLURM_PROCID --num_tasks $NUM_TASKS --batch_size 512
