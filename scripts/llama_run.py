import json
from requests import Session
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
import time

import yaml

from interfaces.mixtral_interface import Mixtral_Interface
from interfaces.document_processor_interface import DGX3LlamaDocumentProcessor, MixtralDocumentProcessor
from interfaces.llama_interface_dgx3 import Llama_Interface
from utils.app_config import AppConfig
from utils.batch_process import BatchProcessor

import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

class MainProcessor:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
        # self.data_file = app_config.data_file
        # self.output_file = app_config.output_file
        # self.rest_endpoint = app_config.rest_endpoint
        # self.max_words = app_config.max_words
    
    def run(self):
        data = load_dataset('json', data_files=[self.app_config.data_file], split="train")
        llama_service = Llama_Interface(session=Session(), rest_endpoint=self.app_config.rest_endpoint)
        # Choose the appropriate processor
        document_processor = DGX3LlamaDocumentProcessor(llama_service,self.app_config)  # or MixtralDocumentProcessor(llm_service)

        #Using words count probably isnt the best, on the otherhand it avoids a tokenization step 
        batch_processor = BatchProcessor(document_processor,max_words= self.app_config.max_words)
        
        
        #batch_processor = BatchProcessor(mixtral_servdice)
        batches = batch_processor.create_batches(data)

        results = []
        pbar = tqdm(total=len(data))

        for batch in batches:
            pbar.set_description(f"Current batch size: {len(batch)}")  # Update the progress bar's description
            batch_processor.process_batch(batch=batch, results=results, pbar=pbar)
           
        results.sort(key=lambda x: x[0])

        with open(self.output_file, 'w') as f:
            json.dump(results, f)

        pbar.close()


# Function to read the yaml file and load it into a Config object
def load_config(yaml_file):
    with open(yaml_file, 'r') as file:
        config_data = yaml.safe_load(file)

    return AppConfig(**config_data)


if __name__ == "__main__":
    app_config = load_config('config/app_config.yaml')
    print(app_config)
    processor = MainProcessor(app_config = app_config)
    processor.run()
