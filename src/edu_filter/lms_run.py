import json
import hydra
from omegaconf import DictConfig
from requests import Session
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
import sys 
import os

sys.path.append(os.path.join(os.getcwd(), 'src'))
from interfaces.document_processor_interface import DocumentProcessor
from edu_filter.llm_api.llm_service import LLMService
from utils.batch_process import BatchProcessor
from utils.app_config import AppConfig


import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

class MainProcessor:
    def __init__(self, app_config: AppConfig):
        self.app_config = app_config
    
    def run(self):
        data = load_dataset('json', data_files=[self.app_config.data_file], split="train")

        llama_service = LLMService(session=Session(), rest_endpoint=self.app_config.rest_endpoint,
                                    app_config=self.app_config)

        document_processor = DocumentProcessor(llama_service,self.app_config)  

        #Using words count probably isnt the best, on the otherhand it avoids a tokenization step 
        batch_processor = BatchProcessor(document_processor,max_words= self.app_config.max_words)
        
        batches = batch_processor.create_batches(data)

        results = []
        pbar = tqdm(total=len(data))

        for batch in batches:
            pbar.set_description(f"Current batch size: {len(batch)}")  # Update the progress bar's description
            batch_processor.process_batch(batch=batch, results=results, pbar=pbar)
           
        results.sort(key=lambda x: x[0])

        with open(self.app_config.output_file, 'w') as f:
            json.dump(results, f)

        pbar.close()


# Loading config with hydra
def run_hydra(app_config):
    @hydra.main(config_path="/raid/s3/opengptx/mehdi/git_repos/edu_filter/scripts/config", config_name="app_config")
    def hydra_entry(cfg : DictConfig) -> None:
        app_config.load_config(cfg)  # Call the load_config method
    hydra_entry()


if __name__ == "__main__":

    app_config = AppConfig()
    run_hydra(app_config)
    processor = MainProcessor(app_config = app_config)
    processor.run()
