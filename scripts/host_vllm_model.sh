set -x
# set cwd to the parent path of this file
cd "$(dirname "$0")"

# load HF_TOKEN from .env file
export $(cat ../.env | xargs)
if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN is not defined. Specify it in your .env file or in your current shell and execute this script with 'source' command."
    exit 1
fi

if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    echo "CUDA_VISIBLE_DEVICES is not set. Specify it in your .env file or in your current shell and execute this script with 'source' command."
    exit 1
fi

if [ -z "$HF_HOME" ]; then
    echo "HF_HOME is not set. Specify it in your .env file or in your current shell and execute this script with 'source' command."
    exit 1
fi

if [ -z "$1" ]; then
    echo "Container name is not provided. Please provide it as the first argument."
    exit 1
fi

if [ -z "$2" ]; then
    echo "Port is not provided. Please provide it as the second argument."
    exit 1
fi

if [ -z "$3" ]; then
    echo "Model name is not provided. Please provide it as the third argument."
    exit 1
fi

CONTAINER_NAME=$1
PORT=$2
MODEL_NAME=$3

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
devices="\"device=$CUDA_VISIBLE_DEVICES\""
docker run \
    --gpus $devices \
    --name $CONTAINER_NAME \
    --rm \
    -v $HF_HOME:/root/.cache/huggingface \
    --env "HF_TOKEN=$HF_TOKEN" \
    -p $PORT:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.7.3 \
    --model $MODEL_NAME \
    --tensor-parallel-size $num_gpus

