from llm_interface.interface.interface import LanguageModelAPI,InputTextToLong
import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from requests import RequestException, Session
import json
import time
from llm_interface.dto.model_config import ModelName
from llm_interface.dto.decoding_strategy import TopPDecodingStrategy, TopKDecodingStrategy, TypicalPDecodingStrategy, BeamSearchDecodingStrategy
from llm_interface.dto.prompt_response import PromptResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from requests.adapters import HTTPAdapter


class Mixtral_Interface():
    """
    This is a convenience class. It creates a REST call to the LanguageModelService given the endpoint below.
    """

    def __init__(self, session: Session, rest_endpoint: str):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__session = session
        #Not entirely sure why this is needed now, but it worked fine previously
        self.__session.mount('http://', HTTPAdapter(pool_connections=100, pool_maxsize=100))
        self.__rest_endpoint_generate = f"{rest_endpoint}generate" if rest_endpoint.endswith("/") else f"{rest_endpoint}/generate"
        self.__rest_endpoint_tokenize = f"{rest_endpoint}tokenize" if rest_endpoint.endswith("/") else f"{rest_endpoint}/tokenize"
        self.__rest_endpoint_detokenize = f"{rest_endpoint}detokenize" if rest_endpoint.endswith("/") else f"{rest_endpoint}/detokenize"
        self.__rest_endpoint_available_models = f"{rest_endpoint}models" if rest_endpoint.endswith("/") else f"{rest_endpoint}/models"

        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_generate}")
        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_tokenize}")
        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_detokenize}")
        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_available_models}")

    def generate(self,
                 prompt: Union[str, List[int]],
                 decoding: Union[TopPDecodingStrategy, TopKDecodingStrategy, TypicalPDecodingStrategy, BeamSearchDecodingStrategy],
                 model_name: ModelName,
                 openai_key: Optional[str] = None,
                 request_id: Optional[str] = None):

        if request_id is None:
            request_id = str(uuid.uuid4())

        max_retries = 2
        backoff_factor = 0.4 

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x22B-Instruct-v0.1")
        #Apply the prompt template
        inputs = tokenizer.apply_chat_template(prompt, tokenize=False)

        content=dict({"inputs": inputs,"model": model_name, "parameters": dict( details=True, 
                    max_tokens= decoding["max_tokens"],max_new_tokens=decoding["max_new_tokens"],
                    temperature=decoding["temperature"],)})
        
        for i in range(max_retries):
            try:
                response = self.__session.post(
                    url=self.__rest_endpoint_generate,
                    json=content,
                    timeout=20, #wait for 10 seconds max 
                
                )
                #print(f"response is {response.text}")
                return response.json()
            except RequestException as e:
                    print(f"Request failed with {e}, retrying...{i}")
                    time.sleep(backoff_factor * (2 ** i))
                    #print(f"the prompt length was {prompt}")
        return None