MLFilter is a versatile, lightweight framework designed to facilitate the training of machine learning-based filters, particularly for identifying and curating high-quality datasets such as educational content.

Key Features:

- LLMService: This service provides seamless access to hosted large language models (LLMs) that evaluate the quality of documents using custom, user-defined prompts. By leveraging powerful LLMs, MLFilter enables the creation of training dataset for classifiers that filter documents based on their quality.

- Modalities: MLFilter includes an entry point for Modalities, allowing users to train and fine-tune classifiers based on the generated datasets. This feature enables the creation of specialized models tailored to specific needs and domains, enhancing the utility of the framework for a wide range of applications.
    
## Setup LLMService on DGX-01:
  1. Export the model weights path: `export HUGGINGFACECACHE=/raid/data/checkpoints/data`
  2. Export your huggingface key: `export API_TOKEN=...`
  3. Export the model name: `export MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct`
  4. Run `docker run -d --gpus all --shm-size 1g -p 8090:80 -v ${HUGGINGFACECACHE}:/data -e HF_TOKEN=$API_TOKEN ghcr.io/huggingface/text-generation-inference:2.2.0 --model-id $MODEL_NAME --num-shard 8 --max-input-length 65535 --max-total-tokens 65536 --max-batch-prefill-tokens 66536`
  5. Instead of using `--gpus all`, you can restrict the number of GPUs by defining the visivle devices, e.g., '"device=0,1,2,3”. If you do so, oyu ned to defined `--num-shard 4' as well
     

## How To Score Documents:

1. Define a model you want to run:
   In the config/default values you can choose between mixtral and llama
2. If you running your script from outside the machine which is hosting the LLM, create an ssh tunnel `ssh -o ServerAliveInterval=60 -L 8090:localhost:8090 user-name@85.215.1.201`   
3. You can find the rest of other important parameters like data_file and temperature in the same file.
4. Finall from script directly run: `python -m src.lms_run`

## Batching and TGI containers
![image](https://github.com/user-attachments/assets/9f4673a2-5556-489d-b65b-458d2ec8f22e)

TGI internally uses a buffer and performs dynamic batching. To make sure we get the maximum numbers documents processed per request, As a work around, we create batches, where each batch is close to the the capcity of the buffer size and than run .generate via multiple threading. 
