import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from omegaconf import DictConfig
from requests import RequestException, Session
import time

from transformers import AutoTokenizer
from requests.adapters import HTTPAdapter

from utils.app_config import AppConfig

class LLMRestClient:
    """
    This is a convenience class. It creates a REST call to the hosted tgi container given the endpoint
    """

    def __init__(self, cfg: DictConfig, session: Session, rest_endpoint: str):
        self.max_retries = cfg.max_retries
        self.backoff_factor = cfg.backoff_factor
        self.model_name = cfg.model_name
        self.max_tokens = cfg.max_tokens
        self.max_new_tokens = cfg.max_new_tokens
        self.temperature = cfg.temperature
        self.timeout = cfg.timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = session
        #Not entirely sure why this is needed now, but it worked fine previously
        self.session.mount('http://', HTTPAdapter(pool_connections=1000, pool_maxsize=1000))
        
        self.rest_endpoint_generate = f"{rest_endpoint}generate" if rest_endpoint.endswith("/") else f"{rest_endpoint}/generate"
        self.logger.info(f"Using rest endpoint at {self.rest_endpoint_generate}")
  
    def generate(
        self,
        prompt: Union[str, List[int]],
        request_id: Optional[str] = None,
        ):

        if request_id is None:
            request_id = str(uuid.uuid4())

        
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        #Apply the prompt template
        inputs = tokenizer.apply_chat_template(prompt, tokenize=False)
        
        content=dict(
            {
                    "inputs": inputs,
                    "model": self.model_name,
                    "parameters": dict(
                        details=True, 
                        max_tokens= self.max_tokens,
                        max_new_tokens= self.max_new_tokens,
                        temperature= self.temperature,
                    )
                }
        )
        
        
        for i in range(self.max_retries):
            try:
                response = self.session.post(
                    url=self.rest_endpoint_generate,
                    json=content,
                    timeout=self.timeout, 
                
                )
                
                return response.json()
            except RequestException as e:
                    print(f"Request failed with {e}, retrying...{i}")
                    time.sleep(self.backoff_factor * (2 ** i))
        return None