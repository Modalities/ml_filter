from llm_interface.interface.interface import LanguageModelAPI,InputTextToLong
import logging
import uuid
from abc import ABC, abstractmethod
from typing import List, Optional, Union

from requests import RequestException, Session
import json

from llm_interface.dto.model_config import ModelName
from llm_interface.dto.decoding_strategy import TopPDecodingStrategy, TopKDecodingStrategy, TypicalPDecodingStrategy, BeamSearchDecodingStrategy
from llm_interface.dto.prompt_response import PromptResponse
from transformers import AutoModelForCausalLM, AutoTokenizer

class Mixtral_Interface(LanguageModelAPI):
    """
    This is a convenience class. It creates a REST call to the LanguageModelService given the endpoint below.
    """

    def __init__(self, session: Session, rest_endpoint: str):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__session = session
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

        tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x22B-Instruct-v0.1")
        #Apply the prompt template
        inputs = tokenizer.apply_chat_template(prompt, tokenize=False)
        #print("##########################")

        #print(f'restend point is {self.__rest_endpoint_generate}')

        content=dict({"inputs": inputs,"model": model_name, "parameters": dict( details=True, 
                    max_tokens= 5000,max_new_tokens=500,
                    temperature=0.001,)})
        try:
            response = self.__session.post(
                url=self.__rest_endpoint_generate,
                json=content,
            
            )
            #print(f"response is {response.text}")
            return response.json()
        except RequestException as e:
                print(f"Request failed with {e}")
                print(f"the prompt length was {prompt}")
                return None

    def tokenize(self,
                 model_name: ModelName,
                 text: str,
                 padding: bool = False,
                 add_special_tokens: bool = False,
                 request_id: Optional[str] = None) -> List[int]:

        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self.__session.post(
            url=self.__rest_endpoint_tokenize,
            params={
                "text": text,
                "model_name": model_name.value,
                "request_id": request_id,
                "padding": padding,
                "add_special_tokens": add_special_tokens
            }
        )

        if response.status_code == 200:
            return response.json()
        elif "Input too long. Max length" in response.text:
            raise InputTextToLong(f"For request id {request_id}, large Language Model Service at "
                                  f"'{self.__rest_endpoint_tokenize}' reported error code "
                                  f"{response.status_code} and message '{response.text}'.")
        else:
            raise ValueError(f"Could not make request to Large Language Model Service at '{self.__rest_endpoint_tokenize}', "
                             f"see status {response.status_code} and message {response.text} for request id {request_id}")

    def detokenize(self, tokens: List[int], model_name: ModelName, request_id: Optional[str] = None) -> str:
        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self.__session.post(
            url=self.__rest_endpoint_detokenize,
            json=tokens,
            params={
                "model_name": model_name.value,
                "request_id": request_id
            }
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Could not make request to Large Language Model Service at '{self.__rest_endpoint_detokenize}', "
                             f"see status {response.status_code} and message {response.text} for request id {request_id}")

    def available_models(self, request_id: Optional[str] = None) -> List[str]:
        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self.__session.get(url=self.__rest_endpoint_available_models)

        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(f"Could not make request to Large Language Model Service at '{self.__rest_endpoint_available_models}', "
                             f"see status {response.status_code} and message {response.text} for request id {request_id}")
