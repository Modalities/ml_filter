from typing import List
import yaml
import hydra
from omegaconf import DictConfig, OmegaConf
import logging

logger = logging.getLogger(__name__)


# Define a class to hold the configuration
class AppConfig:
    def __init__(self):
        self.data_file:str 
        self.output_file:str
        self.rest_endpoint:str
        self.model_name:str
        self.max_tokens:int
        self.max_new_tokens:int
        self.temperature:float
        self.max_words:int
        self.max_words: int
        self.backoff_factor: int
        self.timeout: int
        self.strings_to_remove: List[str]
        self.fineweb_prompt: str

        # Define the model name mapping
        self.model_mapping = {
            "mixtral": "mistralai/Mixtral-8x22B-Instruct-v0.1",
            "llama": "meta-llama/Meta-Llama-3.1-70B-Instruct"
        }

    def __repr__(self):
        return (f"Config(data_file={self.data_file}, output_file={self.output_file}, "
                f"rest_endpoint={self.rest_endpoint}, model_name={self.model_name}, "
                f"max_tokens={self.max_tokens}, max_new_tokens={self.max_new_tokens}, "
                f"max_retries={self.max_retries}, backoff_factor={self.backoff_factor}, "
                f"temperature={self.temperature}, max_words= {self.max_words},"
                f"timeout={self.timeout}, strings_to_remove={self.strings_to_remove}," 
                f"fineweb_prompt= {self.fineweb_prompt})")
    
    def load_config(self, cfg : DictConfig) -> None:
        #logger.info(OmegaConf.to_yaml(cfg))  # Log the configuration data
        self.data_file = cfg.data_file
        self.output_file = cfg.output_file
        self.rest_endpoint = cfg.rest_endpoint
        self.model_name = self.model_mapping.get(cfg.model_name, cfg.model_name)
        self.max_tokens = cfg.max_tokens
        self.max_new_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.max_words =  cfg.max_words  
        self.max_retries = cfg.max_retries
        self.backoff_factor = cfg.backoff_factor
        self.timeout = cfg.timeout
        self.strings_to_remove = cfg.strings_to_remove
        self.fineweb_prompt = cfg.fineweb.prompt
