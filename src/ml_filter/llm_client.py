import logging
import shutil
from pathlib import Path

from omegaconf import OmegaConf
from requests import Session

from ml_filter.config.annotation_pipeline_config import AnnotationPipelineConfig
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

        OmegaConf.register_new_resolver("eval", eval)
        config_omegaconf = OmegaConf.load(config_file_path)
        config_resolved = OmegaConf.to_container(config_omegaconf, resolve=True)
        cfg = AnnotationPipelineConfig.model_validate(config_resolved)

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
        self.raw_data_file_paths = [Path(path) for path in cfg.settings.paths.raw_data_file_paths]

        # LLMRestClient related variables
        self.max_retries = cfg.llm_rest_client.max_retries
        self.backoff_factor = cfg.llm_rest_client.backoff_factor
        self.model_name = cfg.llm_rest_client.model_name
        self.timeout = cfg.llm_rest_client.timeout
        self.max_pool_connections = cfg.llm_rest_client.max_pool_connections
        self.max_pool_maxsize = cfg.llm_rest_client.max_pool_maxsize
        self.max_tokens = cfg.llm_rest_client.max_tokens

        self.verbose = cfg.llm_rest_client.verbose
        self.sampling_params = cfg.llm_rest_client.sampling_params

        # Tokenizer related variables
        self.pretrained_model_name_or_path = Path(cfg.tokenizer.pretrained_model_name_or_path)
        self.special_tokens = cfg.tokenizer.special_tokens
        self.add_generation_prompt = cfg.tokenizer.add_generation_prompt

        # DocumentProcessor related variables
        self.max_prompt_length = cfg.prompt_builder.max_prompt_length
        self.queue_size = cfg.document_processor.queue_size
        self.num_processes = cfg.document_processor.num_processes
        self.score_metric_name = cfg.document_processor.score_metric_name
        self.jq_language_pattern = cfg.document_processor.jq_language_pattern

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
            add_generation_prompt=self.add_generation_prompt,
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
            sampling_params=self.sampling_params,
            verbose=self.verbose,
        )

        # Get DocumentProcessor
        document_processor = DocumentProcessor(
            llm_rest_client=llm_rest_client,
            prompt_builder=PromptBuilder(
                self.prompt_template_file_path, tokenizer=tokenizer, max_prompt_length=self.max_prompt_length
            ),
            queue_size=self.queue_size,
            raw_data_file_paths=self.raw_data_file_paths,
            experiment_dir_path=self.experiment_dir_path,
            num_processes=self.num_processes,
            score_metric_name=self.score_metric_name,
            jq_language_pattern=self.jq_language_pattern,
        )

        document_processor.run()
