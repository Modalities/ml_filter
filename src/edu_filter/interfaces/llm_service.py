import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from requests import RequestException, Session
import time

from transformers import AutoTokenizer
from requests.adapters import HTTPAdapter

from utils.app_config import AppConfig

class Llm_Service():
    """
    This is a convenience class. It creates a REST call to the hosted tgi container given the endpoint
    """

    def __init__(self, session: Session, rest_endpoint: str, app_config: AppConfig):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__session = session
        self.app_config = app_config
        #Not entirely sure why this is needed now, but it worked fine previously
        self.__session.mount('http://', HTTPAdapter(pool_connections=1000, pool_maxsize=1000))
        
        self.__rest_endpoint_generate = f"{rest_endpoint}generate" if rest_endpoint.endswith("/") else f"{rest_endpoint}/generate"
        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_generate}")
  
    def generate(self,
                 prompt: Union[str, List[int]],
                 openai_key: Optional[str] = None,
                 request_id: Optional[str] = None):

        if request_id is None:
            request_id = str(uuid.uuid4())

        max_retries = self.app_config.max_retries
        backoff_factor = self.app_config.backoff_factor

        tokenizer = AutoTokenizer.from_pretrained(self.app_config.model_name)
        #Apply the prompt template
        inputs = tokenizer.apply_chat_template(prompt, tokenize=False)
        
        content=dict({"inputs": inputs,"model": self.app_config.model_name, "parameters": dict( details=True, 
                    max_tokens= self.app_config.max_tokens, max_new_tokens= self.app_config.max_new_tokens,
                    temperature= self.app_config.temperature,)})
        
        
        for i in range(max_retries):
            try:
                response = self.__session.post(
                    url=self.__rest_endpoint_generate,
                    json=content,
                    timeout=self.app_config.timeout, 
                
                )
                
                return response.json()
            except RequestException as e:
                    print(f"Request failed with {e}, retrying...{i}")
                    time.sleep(backoff_factor * (2 ** i))
        return None