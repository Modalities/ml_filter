#!/bin/sh

# Define the remote and local directories
REMOTE_HOST="mn5_transfer"  # Host specified in the SSH config
REMOTE_DIR="/gpfs/projects/ehpc17/weber_alex/models/"
LOCAL_DIR="/raid/s3/opengptx/models/hub"

# List of models to sync
MODELS=(
    "models--Qwen--Qwen2.5-7B-Instruct"
    "models--Qwen--Qwen2.5-72B-Instruct"
    "models--Qwen--Qwen2.5-14B-Instruct"
    "models--Qwen--Qwen2.5-32B-Instruct"
    "models--meta-llama--Llama-3.1-8B-Instruct"
    "models--meta-llama--Llama-3.2-3B-Instruct"
    "models--meta-llama--Llama-3.3-70B-Instruct"
    "models--google--gemma-2-27b-it"
    "models--google--gemma-2-9b-it"
)

# Sync all models in a single rsync command
rsync -avzh --progress "${MODELS[@]/#/${LOCAL_DIR}/}" "${REMOTE_HOST}:${REMOTE_DIR}"
