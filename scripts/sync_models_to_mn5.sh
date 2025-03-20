#!/bin/bash

# Define the remote and local directories
REMOTE_HOST="mn5_transfer"  # Host specified in the SSH config
REMOTE_DIR="/gpfs/projects/ehpc17/models/"
LOCAL_DIR="/raid/s3/opengptx/models/hub"

# List of models to sync
MODELS=(
    "models--mistralai--Mistral-Small-24B-Instruct-2501"
    "models--AtlaAI--Selene-1-Mini-Llama-3.1-8B"
)
# Sync all models in a single rsync command
rsync -avzh --progress "${MODELS[@]/#/${LOCAL_DIR}/}" "${REMOTE_HOST}:${REMOTE_DIR}"
