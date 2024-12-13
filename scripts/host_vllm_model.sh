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

num_gpus=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
devices="\"device=$CUDA_VISIBLE_DEVICES\""
docker run --runtime nvidia \
    --gpus $devices \
    --name alex_vllm_container \
    -v /raid/s3/opengptx/models/:/root/.cache/huggingface \
    --env "HUGGING_FACE_HUB_TOKEN=$HF_TOKEN" \
    -p 9900:8000 \
    --ipc=host \
    vllm/vllm-openai:v0.6.3 \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --tensor-parallel-size $num_gpus

