#!/bin/bash
#SBATCH --job-name=sampling_pipeline
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=barnard
#SBATCH --account=p_gptx
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=100
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=4500

# Load modules and activate environment
module purge

module --ignore_cache load release/24.04
module load GCC/13.2.0
module load Python/3.11.5 

source /data/horse/ws/alju972f-tokenization_at_scale/env_tokenization/bin/activate

LANG=$1
echo "Running pipeline for $LANG"

python3 run_language_pipeline.py $LANG
