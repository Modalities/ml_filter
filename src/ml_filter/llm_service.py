from pathlib import Path
from omegaconf import OmegaConf
from requests import Session
from datasets import load_dataset
import sys 
import os

from ml_filter.tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer

sys.path.append(os.path.join(os.getcwd(), 'src'))

from ml_filter.data_processing.document_processor import DocumentProcessor
from ml_filter.llm_api.llm_rest_client import LLMRestClient


import logging

logging.getLogger("transformers").setLevel(logging.ERROR)

class LLMService:
    def __init__(self, config_file_path:Path):
        """Initializes the LLMService."""
        cfg = OmegaConf.load(config_file_path)
        # Dataset related variables
        self.data_file_path = cfg.data.input_data.path
        self.split = cfg.data.input_data.split
        
        # LLMRestClient related variables
        self.output_file_path = cfg.data.output_data_path
        self.rest_endpoint = cfg.rest_endpoint
        self.max_retries = cfg.max_retries
        self.backoff_factor = cfg.backoff_factor
        self.model_name = cfg.model_name
        self.timeout = cfg.timeout
        self.max_pool_connections = cfg.max_pool_connections
        self.max_pool_maxsize = cfg.max_pool_maxsize

        # Tokenizer related variables
        self.pretrained_model_name_or_path = cfg.tokenizer.pretrained_model_name_or_path
        self.truncation = cfg.tokenizer.truncation
        self.padding = cfg.tokenizer.padding
        self.max_length = cfg.tokenizer.max_length
        self.special_tokens = cfg.tokenizer.special_tokens

        # DocumentProcessor related variables
        self.prompt_template = cfg.document_processor.prompt_template
        self.queue_size = cfg.document_processor.queue_size
        self.batch_size = cfg.document_processor.batch_size
        self.max_tokens = cfg.document_processor.max_tokens
        self.max_new_tokens = cfg.document_processor.max_new_tokens
        self.temperature = cfg.document_processor.temperature
        self.verbose = cfg.document_processor.verbose
    
    def run(self):
        """Runs the LLM service.
        
        This method loads the dataset, initializes the tokenizer, LLMRestClient, and DocumentProcessor,
        and then runs the document processing on the loaded data to obtain the model responses.
        """

        # Get data
        data = load_dataset('json', data_files=[self.data_file_path], split=self.split)
        
        # Get Tokenizer
        tokenizer = PreTrainedHFTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            truncation=self.truncation,
            padding=self.padding,
            max_length=self.max_length,
            special_tokens=self.special_tokens,
        )

        # Get LLMRestClient
        llm_rest_client = LLMRestClient(
            max_retries=self.max_retries,
            backoff_factor=self.backoff_factor,
            model_name=self.model_name,
            timeout=self.timeout,
            session=Session(), 
            rest_endpoint=self.rest_endpoint,
            tokenizer=tokenizer,
            max_pool_connections=self.max_pool_connections,
            max_pool_maxsize=self.max_pool_maxsize,
        )

        # Get DocumentProcessor
        document_processor = DocumentProcessor(
            llm_rest_client=llm_rest_client,
            prompt_template=self.prompt_template,
            queue_size=self.queue_size,
            batch_size=self.batch_size,
            max_tokens=self.max_tokens,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            verbose=self.verbose,
            output_file_path=self.output_file_path,
        )
        
        document_processor.run(data)

