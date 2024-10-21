import logging
import os
import sys
from pathlib import Path

from datasets import load_dataset
from omegaconf import OmegaConf
from requests import Session

from ml_filter.data_processing.document_processor import DocumentProcessor
from ml_filter.data_processing.prompt_builder import PromptBuilder
from ml_filter.llm_api.llm_rest_client import LLMRestClient
from ml_filter.tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer

sys.path.append(os.path.join(os.getcwd(), "src"))


logging.getLogger("transformers").setLevel(logging.ERROR)


class LLMClient:
    def __init__(self, config_file_path: Path):
        """Initializes the LLMService."""
        cfg = OmegaConf.load(config_file_path)
        # Dataset related variables
        self.data_file_path = cfg.data.input_data.path
        self.split = cfg.data.input_data.split

        # LLMRestClient related variables
        self.rest_endpoint = cfg.llm_rest_client.rest_endpoint
        self.max_retries = cfg.llm_rest_client.max_retries
        self.backoff_factor = cfg.llm_rest_client.backoff_factor
        self.model_name = cfg.llm_rest_client.model_name
        self.timeout = cfg.llm_rest_client.timeout
        self.max_pool_connections = cfg.llm_rest_client.max_pool_connections
        self.max_pool_maxsize = cfg.llm_rest_client.max_pool_maxsize
        self.max_tokens = cfg.llm_rest_client.max_tokens
        self.max_new_tokens = cfg.llm_rest_client.max_new_tokens
        self.temperature = cfg.llm_rest_client.temperature
        self.verbose = cfg.llm_rest_client.verbose

        # Tokenizer related variables
        self.pretrained_model_name_or_path = cfg.tokenizer.pretrained_model_name_or_path
        self.special_tokens = cfg.tokenizer.special_tokens

        # DocumentProcessor related variables
        self.output_file_path = cfg.document_processor.output_file_path
        self.prompt_template_path = cfg.prompt_builder.prompt_path
        self.queue_size = cfg.document_processor.queue_size
        self.batch_size = cfg.document_processor.batch_size
        self.num_processes = cfg.document_processor.num_processes
        self.score_metric_name = cfg.document_processor.score_metric_name

    def run(self):
        """Runs the LLM service.

        This method loads the dataset, initializes the tokenizer, LLMRestClient, and DocumentProcessor,
        and then runs the document processing on the loaded data to obtain the model responses.
        """

        # Get data
        data = load_dataset("json", data_files=[self.data_file_path], split=self.split)

        # Get Tokenizer
        tokenizer = PreTrainedHFTokenizer(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            truncation=False,
            padding=False,
            max_length=None,
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
            max_tokens=self.max_tokens,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            verbose=self.verbose,
        )

        # Get DocumentProcessor
        document_processor = DocumentProcessor(
            llm_rest_client=llm_rest_client,
            prompt_builder=PromptBuilder(self.prompt_template_path),
            queue_size=self.queue_size,
            batch_size=self.batch_size,
            output_file_path=self.output_file_path,
            num_processes=self.num_processes,
            score_metric_name=self.score_metric_name,
        )

        document_processor.run(data)
