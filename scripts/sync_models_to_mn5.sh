#!/bin/bash

# Define the remote and local directories
REMOTE_HOST="mn5_transfer"  # Host specified in the SSH config
REMOTE_DIR="/gpfs/projects/ehpc17/models/"
LOCAL_DIR="/home/user-name/.cache/huggingface/hub"

# List of models to sync
MODELS=(
    "models--mistralai--Mistral-Small-3.1-24B-Instruct-2503"
)
# Sync all models in a single rsync command
rsync -avzh --progress "${MODELS[@]/#/${LOCAL_DIR}/}" "${REMOTE_HOST}:${REMOTE_DIR}"
