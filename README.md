MLFilter is a versatile, lightweight framework designed to facilitate the training of machine learning-based filters, particularly for identifying and curating high-quality datasets such as educational content.

Key Features:

- Dataset Generation: A client provides seamless access to hosted large language models (LLMs) that evaluate the quality of documents using custom, user-defined prompts. By leveraging powerful LLMs, MLFilter enables the creation of training dataset for classifiers that filter documents based on their quality.

- Training of Classifiers: MLFilter provides training functionalities allowing users to train and fine-tune classifiers based on the generated datasets. This feature enables the creation of specialized models tailored to specific needs and domains, enhancing the utility of the framework for a wide range of applications.

## Running the TGI Container with Hugging Face Models

This service relies on **TGI containers** (Text Generation Inference), which can be downloaded from [Hugging Face](https://huggingface.co). Follow the steps below to download and run the TGI container.

### 1. Set Up Environment Variables

First, you'll need to export some environment variables for the model's download path, Hugging Face API key, and the model's full name.

1. **Set the model cache directory:**

   Define the path where the model weights will be downloaded or where they already exist:
   ```bash
   export HUGGINGFACECACHE=/raid/data/checkpoints/data
2. **Export your Hugging Face API token:**

   You need an API token from Hugging Face. Replace ... with your actual token:
   ```bash
   export API_TOKEN=your_huggingface_api_token_here
   
3. **Specify the model name:**

   Provide the full name of the model as it appears on Hugging Face (e.g., meta-llama/Llama-3.1-70B-Instruct):
   ```bash
   export MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct

### 2. Download and Run the TGI Container

Use the following command to download the TGI container and run it. If the model weights are already in the specified path, the download step will be skipped.
        
        docker run -d --gpus all --shm-size 1g -p 8090:80 \
        -v ${HUGGINGFACECACHE}:/data \
        -e HF_TOKEN=$API_TOKEN \
        ghcr.io/huggingface/text-generation-inference:2.2.0 \
        --model-id $MODEL_NAME \
        --num-shard 8 \
        --max-input-length 65535 \
        --max-total-tokens 65536 \
        --max-batch-prefill-tokens 66536
 ### 3. Optional: Restricting GPU Usage     
 By default, the container uses all available GPUs (--gpus all). If you want to limit the number of GPUs, you can define specific devices. For example, to restrict the container to 4 GPUs (e.g.,  devices 0, 1, 2, 3), use the following:
 
     docker run -d --gpus '"device=0,1,2,3"' --shm-size 1g -p 8090:80 \
    -v ${HUGGINGFACECACHE}:/data \
    -e HF_TOKEN=$API_TOKEN \
    ghcr.io/huggingface/text-generation-inference:2.2.0 \
    --model-id $MODEL_NAME \
    --num-shard 4 \
    --max-input-length 65535 \
    --max-total-tokens 65536 \
    --max-batch-prefill-tokens 66536

Make sure to update --num-shard to match the number of GPUs you're using.



## How To Score Documents:

1. Define a model you want to run:
   In the config/default values you can choose between mixtral and llama
2. If you running your script from outside the machine which is hosting the LLM, create an ssh tunnel `ssh -o ServerAliveInterval=60 -L 8090:localhost:8090 user-name@85.215.1.201`   
3. You can find the rest of other important parameters like data_file and temperature in the same file.
4. Finall from script directly run: `python -m src.lms_run`

## Batching and TGI containers
![image](https://github.com/user-attachments/assets/9f4673a2-5556-489d-b65b-458d2ec8f22e)

TGI internally uses a buffer and performs dynamic batching. To make sure we get the maximum numbers documents processed per request, As a work around, we create batches, where each batch is close to the the capcity of the buffer size and than run .generate via multiple threading. 
