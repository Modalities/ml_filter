import logging
import shutil
from pathlib import Path

from omegaconf import OmegaConf
from requests import Session

from ml_filter.data_processing.document_processor import DocumentProcessor
from ml_filter.data_processing.prompt_builder import PromptBuilder
from ml_filter.llm_api.llm_rest_client import LLMRestClient
from ml_filter.tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer

logging.getLogger("transformers").setLevel(logging.ERROR)


class LLMClient:
    def __init__(self, config_file_path: Path, experiment_id: str, rest_endpoint: str):
        """Initializes the LLMService."""
        self.experiment_id = experiment_id
        self.rest_endpoint = rest_endpoint

        cfg = OmegaConf.load(config_file_path)
        self.inference_server_type = cfg.llm_rest_client.host_type
        inference_server_type = ["vllm", "tgi"]
        assert (
            self.inference_server_type in inference_server_type
        ), f"Invalid host type: {self.inference_server_type} must be in {inference_server_type}"

        self.prompt_template_file_path = Path(cfg.prompt_builder.prompt_template_file_path)
        # Create experiment directory and store the config as backup
        self.experiment_dir_path = Path(cfg.settings.paths.output_directory_path) / self.experiment_id
        self.experiment_dir_path.mkdir(parents=True, exist_ok=True)
        shutil.copy(config_file_path, self.experiment_dir_path / config_file_path.name)
        shutil.copy(
            cfg.prompt_builder.prompt_template_file_path,
            self.experiment_dir_path / Path(self.prompt_template_file_path).name,
        )
        # Dataset related variables
        self.raw_data_file_path = Path(cfg.settings.paths.raw_data_file_path)

        # LLMRestClient related variables
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
        self.pretrained_model_name_or_path = Path(cfg.tokenizer.pretrained_model_name_or_path)
        self.special_tokens = cfg.tokenizer.special_tokens

        # DocumentProcessor related variables
        self.max_prompt_length = cfg.prompt_builder.max_prompt_length
        self.queue_size = cfg.document_processor.queue_size
        self.batch_size = cfg.document_processor.batch_size
        self.num_processes = cfg.document_processor.num_processes
        self.score_metric_name = cfg.document_processor.score_metric_name

    def run(self):
        """Runs the LLM service.

        This method loads the dataset, initializes the tokenizer, LLMRestClient, and DocumentProcessor,
        and then runs the document processing on the loaded data to obtain the model responses.
        """

        # Get Tokenizer
        # This tokenizer is only used for applying the chat template, but is not applied within TGI.
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
            max_pool_connections=self.max_pool_connections,
            max_pool_maxsize=self.max_pool_maxsize,
            max_tokens=self.max_tokens,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            verbose=self.verbose,
            inference_server_type=self.inference_server_type,
        )

        # Get DocumentProcessor
        document_processor = DocumentProcessor(
            llm_rest_client=llm_rest_client,
            prompt_builder=PromptBuilder(
                self.prompt_template_file_path, tokenizer=tokenizer, max_prompt_length=self.max_prompt_length
            ),
            queue_size=self.queue_size,
            batch_size=self.batch_size,
            raw_data_file_path=self.raw_data_file_path,
            experiment_dir_path=self.experiment_dir_path,
            num_processes=self.num_processes,
            score_metric_name=self.score_metric_name,
        )

        document_processor.run()
