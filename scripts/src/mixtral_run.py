import json
from requests import Session
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from datasets import load_dataset
import time

from interfaces.mixtral_interface import Mixtral_Interface
from interfaces.document_processor_interface import MixtralDocumentProcessor
from utils.batch_process import BatchProcessor



class MainProcessor:
    def __init__(self, data_file: str, output_file: str, rest_endpoint: str):
        self.data_file = data_file
        self.output_file = output_file
        self.rest_endpoint = rest_endpoint

    def run(self):
        data = load_dataset('json', data_files=[self.data_file], split="train[:5%]")
        mixtral_service = Mixtral_Interface(session=Session(), rest_endpoint=self.rest_endpoint)
         # Choose the appropriate processor
        document_processor = MixtralDocumentProcessor(mixtral_service)  # or LlamaDocumentProcessor(llm_service)

        #max_length + max_new_token > 4096 tokens throws an error
        batch_processor = BatchProcessor(document_processor,max_words=30000)
        
        
        #batch_processor = BatchProcessor(mixtral_servdice)
        batches = batch_processor.create_batches(data)

        results = []
        pbar = tqdm(total=len(data))

        for batch in batches:
            #print(f"Batch size is {len(batch)}")
            batch_processor.process_batch(batch=batch, results=results, pbar=pbar)
           
        results.sort(key=lambda x: x[0])

        with open(self.output_file, 'w') as f:
            json.dump(results, f)

        pbar.close()

if __name__ == "__main__":
    processor = MainProcessor(
        data_file='/data/akhan/fineweb-tsts/oscar_de_filtered_deduplicated_500k/combined_de_500k.jsonl',
        output_file='outputs/mixtral_results_5_percent.json',
        rest_endpoint='http://0.0.0.0:8090/'
    )
    processor.run()
