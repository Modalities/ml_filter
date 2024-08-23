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

    def __repr__(self):
        return (f"Config(data_file={self.data_file}, output_file={self.output_file}, "
                f"rest_endpoint={self.rest_endpoint}, model_name={self.model_name}, "
                f"max_tokens={self.max_tokens}, max_new_tokens={self.max_new_tokens}, "
                f"temperature={self.temperature}, max_words= {self.max_words})")
    
    def load_config(self, cfg : DictConfig) -> None:
        #logger.info(OmegaConf.to_yaml(cfg))  # Log the configuration data
        self.data_file = cfg.data_file
        self.output_file = cfg.output_file
        self.rest_endpoint = cfg.rest_endpoint
        self.model_name = cfg.model_name
        self.max_tokens = cfg.max_tokens
        self.max_new_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.max_words =  cfg.max_words  
        self.fineweb_prompt = cfg.fineweb.prompt
