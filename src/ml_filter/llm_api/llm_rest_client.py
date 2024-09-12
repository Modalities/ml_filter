import logging
from typing import Any, Dict, List

from omegaconf import DictConfig
from requests import RequestException, Session
import time

from requests.adapters import HTTPAdapter

from ml_filter.tokenizer.tokenizer_wrapper import TokenizerWrapper


class LLMRestClient:
    """"A class representing a REST client for the LLM service. 
    This class is responsible for sending requests to the LLM service (hosted tgi container given the endpoint) and returning the response."""
  
    def __init__(
        self,
        max_retries: int,
        backoff_factor: int,
        model_name: str,
        timeout: int,
        session: Session,
        rest_endpoint: str,
        tokenizer: TokenizerWrapper,
        max_pool_connections: int,
        max_pool_maxsize: int,
        ):
        """Initializes the LLMRestClient."""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.model_name = model_name
        self.timeout = timeout
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = session
        self.tokenizer = tokenizer
        # TODO: Not entirely sure why this is needed now, but it worked fine previously
        self.session.mount(
             'http://',
            HTTPAdapter(pool_connections=max_pool_connections, pool_maxsize=max_pool_maxsize)
        )
        
        self.rest_endpoint_generate = f"{rest_endpoint}generate" if rest_endpoint.endswith("/") else f"{rest_endpoint}/generate"
        self.logger.info(f"Using rest endpoint at {self.rest_endpoint_generate}")
  
    def generate(
        self,
        prompt: List[Dict[str, str]],
        max_tokens:int,  
        max_new_tokens: int, 
        temperature: float,
        verbose: bool,
    ) -> Dict[str, str] | None:
        """Generates a response based on the given prompt.
        
        Args:
            prompt (str): The prompt to generate a response for.
            max_tokens (int): The maximum number of tokens in the generated response.
            max_new_tokens (int): The maximum number of new tokens in the generated response.
            temperature (float): The temperature value for controlling randomness in the generated response.
            verbose (bool): Whether to include detailed information in the generated response.
        
        Returns:
            Dict[str, Any]: A dictionary containing the generated response.
        
        Raises:
            ValueError: If max_retries is set to 0.
        """

        inputs = self.tokenizer.apply_tokenizer_chat_template(prompt, tokenize=False)
        
        request = dict(
            {
            "inputs": inputs,
            "model": self.model_name,
            "parameters": dict(
                details=verbose,  # TODO: check if this is correct
                max_tokens=max_tokens,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
            )
            }
        )
        
        if self.max_retries == 0:
            raise ValueError("max_retries must be greater than 0.")
        
        for i in range(self.max_retries):
            try:
                response = self.session.post(
                    url=self.rest_endpoint_generate,
                    json=request,
                    timeout=self.timeout, 
                )
                return response.json()
            except RequestException as e:
                    print(f"Request failed with {e}, retrying...{i}")
                    time.sleep(self.backoff_factor * (2 ** i))
        
        print(f"Request failed after {self.max_retries} retries.")