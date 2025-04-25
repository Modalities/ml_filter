MLFilter is a versatile, lightweight framework designed to facilitate the training of machine learning-based filters, particularly for identifying and curating high-quality datasets such as educational content.

Key Features:

- Dataset Generation: A client provides seamless access to hosted large language models (LLMs) that evaluate the quality of documents using custom, user-defined prompts. By leveraging powerful LLMs, MLFilter enables the creation of training dataset for classifiers that filter documents based on their quality.

- Training of Classifiers: MLFilter provides training functionalities allowing users to train and fine-tune classifiers based on the generated datasets. This feature enables the creation of specialized models tailored to specific needs and domains, enhancing the utility of the framework for a wide range of applications.

## Usage in Eurolingua-GPT
In Eurolingua, we use this repository to filter out low-quality documents from the Common Crawl dataset. The filtered dataset is then used to train the Eurolingua GPT model(s). The following diagram illustrates the workflow, that is closely related to [Fineweb-EDU](https://arxiv.org/pdf/2406.17557): 

1. We start with a Common Crawl (CC) subset (e.g., 200,000 documents per language) that we want to score e.g., w.r.t. the amount of educational content. We use an LLM to score these documents based on the instructios defined in a prompt.

![](https://github.com/EuroLingua-GPT/ml_filter/blob/translation_cli/documentation/diagrams/ml_filters_prompt_based_annotation.svg)

2. The scored documents are then used to train a classifier (probably Roberta) that can be used to filter out low-quality / non-educational documents. 

![](https://github.com/EuroLingua-GPT/ml_filter/blob/translation_cli/documentation/diagrams/ml_filters_classifier_training.svg)

3. The classifier is used to filter out low-quality documents from the entire CC dataset. The filtered dataset is then used to train the Eurolingu GPT model(s).

![](https://github.com/EuroLingua-GPT/ml_filter/blob/translation_cli/documentation/diagrams/ml_filters_classifier_based_annotation.svg)


## Installation and Development

Please see [CONTRIBUTING.md](CONTRIBUTING.md)


## Usage
Once you have [setup TGI container](#setting-up-the-tgi-container-with-hugging-face-models), you can proceed to score and the documents and trainer and classifier

### 1. How to Score Documents with LLM
```script
python cli.py score_documents --config_file_path path/to/your/config.yaml

```
### 2. How to Train a Classifier
If you already have the score, you can train a classifier by running
```script
python cli.py train_classifier --config_file_path path/to/your/training_config.yaml
```
### 3. Measure Interrater Reliability
If you have a dataset with scores annotated by multiple annotators, you can compute metrics to measure the interrater reliability with the command interrater_reliability. If you want to compare the scores in a single file (e.g. the human annotated ground truth data), run:
```script
python cli.py interrater_reliability data_annotated.jsonl --output_file_path output.json
```
If you want to compare the scores across different models and files (e.g. when comparing LLM annotated data to ground truth), the scores in each file first have to be aggregated. For that, use the parameter aggregation:
```script
python cli.py interrater_reliability data_annotated_by_model_1.jsonl data_annotated_by_model_2.jsonl --aggregation majority --output_file_path output.json
```
You can create plots for the distribution of annotations and the differences between annotators with
```script
python cli.py plot_scores data_annotated_by_model_1.jsonl data_annotated_by_model_2.jsonl --aggregation majority --output_dir outputs
```
## TGI

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
   export HF_TOKEN=your_huggingface_api_token_here
   ```
3. **Specify the model name:**

   Provide the full name of the model as it appears on Hugging Face (e.g., meta-llama/Llama-3.1-70B-Instruct):
   ```bash
   export MODEL_NAME=meta-llama/Meta-Llama-3.1-70B-Instruct
   ```
### 2. Download and Run the TGI Container

Use the following command to download the TGI container and run it. If the model weights are already in the specified path, the download step will be skipped.
   
   ```shell  
  docker run -d --gpus all --shm-size 1g -p 8090:80 \
  -v ${HUGGINGFACECACHE}:/data \
  -e HF_TOKEN=$HF_TOKEN \
  ghcr.io/huggingface/text-generation-inference:2.2.0 \
  --model-id $MODEL_NAME \
  --num-shard 8 \
  --max-input-length 65535 \
  --max-total-tokens 65536 \
  --max-batch-prefill-tokens 66536
   ```

If you restrict the number of GPUs for your container by `--gpus '"device=6"'`,  `number-shard` should not be larger.

 ### 3. Optional: Restricting GPU Usage     
 By default, the container uses all available GPUs (--gpus all). If you want to limit the number of GPUs, you can define specific devices. For example, to restrict the container to 4 GPUs (e.g.,  devices 0, 1, 2, 3), use the following:
 
 ```shell
 docker run -d --gpus '"device=0,1,2,3"' --shm-size 1g -p 8090:80 \
 -v ${HUGGINGFACECACHE}:/data \
 -e HF_TOKEN=$API_TOKEN \
 ghcr.io/huggingface/text-generation-inference:2.2.0 \
 --model-id $MODEL_NAME \
 --num-shard 4 \
 --max-input-length 65535 \
 --max-total-tokens 65536 \
 --max-batch-prefill-tokens 66536
```
Make sure to update --num-shard to match the number of GPUs you're using.

### 4. Testing the docker setup
Locate the your container, it will be named  ghcr.io/huggingface/text-generation-inference:2.2.0 
```shell
docker ps
```
You can now look into the logs
```shell
docker logs --follow your_container_id 
```
please note that tgi takes a little bit of time to start

### 5. Testing TGI service
Once the container has been successfully setup and started you can test by running 
```bash
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

## VLLM

#### Host a Model with VLLM (faster)
```bash
docker run --runtime nvidia --gpus '"device=5,6"'  --name vllm_container -v /raid/s3/opengptx/models/:/root/.cache/huggingface --env "HUGGING_FACE_HUB_TOKEN=$API_TOKEN" -p 9900:8000 --ipc=host vllm/vllm-openai:v0.6.3 --model Qwen/Qwen2.5-72B-Instruct-AWQ --tensor-parallel-size 2
```

Number of `tensor-parallel-size` and number of GPUs used should match (`--gpus`).

Alternativley, just use execute `bash scripts/host_vllm_model.sh $CONTAINER_NAME $PORT $MODEL_NAME` and make sure, all required environment variables are in your `.env` file under project root e.g.
```bash
bash scripts/host_vllm_model.sh my_vllm_container 9123 meta-llama/Llama-3.1-8B-Instruct
```

#### Test the hosted model
```bash
curl http://localhost:port_number/v1/completions \
-H "Content-Type: application/json" \
-d '{
"model": "Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4",
"prompt": "San Francisco is a",
"max_tokens": 7,
"temperature": 0
}'
```

#### Look into metrics of the hosted model

1. Forward port 8000 
2. visit http://localhost:8000/metrics to see the tokens/s

Or watch the output e.g. with `Qwen/Qwen2.5-32B-Instruct-GPTQ-Int4` on two GPUs:
```
INFO:ml_filter.data_processing.document_processor:Results written final: 511 | Elapsed time: 215.89 seconds | Results per second: 2.22
```

#### Troubleshooting

> Request failed with HTTPConnectionPool(host='localhost', port=9900): Read timed out. (read timeout=20), retrying...0

With larger models increase the `llm_rest_client.timeout` config parameter.
Also play around with:
```
llm_rest_client.max_pool_connections: 1000
llm_rest_client.max_pool_maxsize: 1000
```


> [VLLM] is already dead, terminating server process.

Solustion as by https://github.com/vllm-project/vllm/issues/10024
```
export VLLM_RPC_TIMEOUT= 20000
```

## Batching and TGI containers
![image](https://github.com/user-attachments/assets/9f4673a2-5556-489d-b65b-458d2ec8f22e)

TGI internally uses a buffer and performs dynamic batching. To make sure we get the maximum numbers documents processed per request, As a work around, we create batches, where each batch is close to the the capcity of the buffer size and than run .generate via multiple threading. 


## Config Advise

add_generation_prompt (bool):  If this is set, a prompt with the token(s) that indicate
                the start of an assistant message will be appended to the formatted output.
                This is useful when you want to generate a response from the model.
                Note that this argument will be passed to the chat template, and so it must be supported in the
                template for this argument to have any effect.
We expect it to work best, if set to true.

## Training an annotation model

Example call for running the training pipeline:
```
ml_filter annotator_training_pipeline --config_file_path path/to/your/config/file.yaml
```
Example configs can be found in [configs/train_classifier](configs/train_classifier/). The most important settings are:
| Setting                              | Description                                                                                                                      |
|--------------------------------------|----------------------------------------------------------------------------------------------------------------------------------|
| `model.name`                         | The name of the base model to be used for training. Only models specified in [src/constants.py](src/constants.py) are supported. |
| `model.freeze_base_model_parameters` | Whether the base model weights should be trained or frozen.                                                                      |
| `...is_regression`                   | Whether a regression head or a classification head should be used.                                                               |
| `...num_tasks`                       | How many tasks should be trained for (e.g. when scoring for educational and adult content together).                             |
| `...num_targets_per_task`            | A list of length num_tasks specifying the number of targets/classes/score values for each task.                                  |
| `data.train_file_path`               | Path to the training dataset file.                                                                                               |
| `data.val_file_path`                 | Path to the validation dataset file.                                                                                             |
| `training.learning_rate`             | The learning rate for the optimizer during training.                                                                             |
| `training.batch_size`                | The number of samples per batch during training.                                                                                 |
| `training.num_epochs`                | The total number of epochs for training the model.                                                                               |
| `training.output_dir_path`             | Directory where the trained model and logs will be saved.                 |

The data should consist of a .jsonl file or a directory structure containing such. Each JSON line should contain a text field matching the data.text_column key from the config. Its value should be a string. Additionally, a scores key specified by data.label_column should contain a list of length num_tasks with the i-th value being in `0, ..., num_targets_per_task[i] - 1`.

The final model will be saved in the `final` subdirectory of `training.output_dir_path`. It will be a transformers style PreTrainedModel and can be loaded via `AnnotatorModel.from_pretrained(".../final")`. Additionally, a copy of the tokenizer of the base model will be saved in the same directory and can be loaded via `AutoTokenizer.from_pretrained(".../final")`.
