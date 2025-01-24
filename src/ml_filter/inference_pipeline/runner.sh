#!/bin/bash
#SBATCH --job-name=bert_inference_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --gres gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=logs/inference_%j.out
#SBATCH --error=logs/inference_%j.err

source ../../../../venvs/ml_filter_build_ba236af2408b6e90856c4ef8d654e7ab7a4fbc49/bin/activate

mkdir -p logs checkpoints outputs

INPUT_FILES_LIST="input_files_list.txt"

srun python inference.py \
  --input_files_list $INPUT_FILES_LIST \
  --output_dir outputs \
  --checkpoint_dir checkpoints \
  --task_id $SLURM_PROCID \
  --num_tasks $SLURM_NTASKS \
  --model_checkpoint ../../../tests/cache/checkpoint-1 \
  --model_arch "snowflake-arctic-embed" \
  --num_regressor_outputs 4 \
  --num_classes_per_output 6 6 2 2 \
  --use_regression True
