import json
from pathlib import Path
from omegaconf import OmegaConf
from requests import Session
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import List, Tuple, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

from edu_filter.llm_api.lms_llama_interface import Llama_Interface_LMS
from interfaces.document_processor_interface import LmsLlamaDocumentProcessor

from utils.batch_process import BatchProcessor

class Main:
    def __init__(self, config_file_path:Path):
        cfg = OmegaConf.load(config_file_path)
        self.data_file_path = cfg.data.input_data.path
        self.split = cfg.data.input_data.split
        self.output_file_path = cfg.data.output_data_path
        self.rest_endpoint = cfg.rest_endpoint

    def run(self):
        data = load_dataset('json', data_files=[self.data_file_path], split=self.split)
        llama_service = Llama_Interface_LMS(session=Session(), rest_endpoint=self.rest_endpoint)
        document_processor = LmsLlamaDocumentProcessor(llama_service) # type: ignore

        #max_length is just my estimation, total words * 1.5 ~ 2 < context length (2048, 4096) 
        # TODO double check the models context length 
        batch_processor = BatchProcessor(document_processor,max_words=60000)
        user_prompt = """Below is an extract from a web page. Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:\n1. Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.\n2. Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.\n3. Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.\n4. Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren’t too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.\n5. Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.\nThe extract: {extract}.\nAfter examining the extract:\nBriefly justify your total score, up to 100 words.\nConclude with the score using the format: 'Educational score: < total points>'"""
        
        batches = batch_processor.create_batches(data)

        results = []
        pbar = tqdm(total=len(data))

        for batch in batches:
            batch_processor.process_batch(batch=batch, results=results, pbar=pbar)

        results.sort(key=lambda x: x[0])

        with open(self.output_file_path, 'w') as f:
            json.dump(results, f)

        pbar.close()

