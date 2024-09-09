## Setup for LLM Service on DGX-01:
  1. export the path where the model weights are located `export HUGGINGFACECACHE=/raid/data/checkpoints/data`
  2. export your huggingface key `export API_TOKEN=...`
  3. export model name `export meta-llama/Meta-Llama-3.1-70B-Instruct`
  4. Finally run `docker run -d --gpus all --shm-size 1g -p 8090:80 -v ${HUGGINGFACECACHE}:/data -e HF_TOKEN=$API_TOKEN ghcr.io/huggingface/text-generation-inference:2.2.0 --model-id $MODEL_NAME --num-shard 8 --max-input-length 65535 --max-total-tokens 65536 --max-batch-prefill-tokens 66536`
  5. Instead of --gpus all you can also run '"device=1,2,3,4” for llama since it does not need all 8 gpus, if you do so please select the --num-shard 4 aswell
     

### How To RUN:

1. Choose a model you want to run:
   In the config/default values you can choose between mixtral and llama
2. You can find the rest of other important parameters like data_file and temperature in the same file.
3. Finall from script directly run: `python -m src.lms_run`
