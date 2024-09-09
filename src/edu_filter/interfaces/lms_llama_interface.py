
import logging
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union

from requests import Session


import time
from requests.exceptions import RequestException

from utils.prompt_templates import Prompt


class Llama_Interface_LMS:
    """
    This is a convenience class. It creates a REST call to the LanguageModelService given the endpoint below.
    """

    def __init__(self, session: Session, rest_endpoint: str):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__session = session
        self.__rest_endpoint_generate = (
            f"{rest_endpoint}generate"
            if rest_endpoint.endswith("/")
            else f"{rest_endpoint}/generate"
        )
        self.__rest_endpoint_tokenize = (
            f"{rest_endpoint}tokenize"
            if rest_endpoint.endswith("/")
            else f"{rest_endpoint}/tokenize"
        )
        self.__rest_endpoint_detokenize = (
            f"{rest_endpoint}detokenize"
            if rest_endpoint.endswith("/")
            else f"{rest_endpoint}/detokenize"
        )
        self.__rest_endpoint_embed = (
            f"{rest_endpoint}embed"
            if rest_endpoint.endswith("/")
            else f"{rest_endpoint}/embed"
        )
        self.__rest_endpoint_explain = (
            f"{rest_endpoint}explain"
            if rest_endpoint.endswith("/")
            else f"{rest_endpoint}/explain"
        )
        self.__rest_endpoint_available_models = (
            f"{rest_endpoint}models"
            if rest_endpoint.endswith("/")
            else f"{rest_endpoint}/models"
        )

        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_generate}")
        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_tokenize}")
        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_detokenize}")
        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_embed}")
        self.__logger.info(f"Using rest endpoint at {self.__rest_endpoint_explain}")
        self.__logger.info(
            f"Using rest endpoint at {self.__rest_endpoint_available_models}"
        )
    

    def generate(
        self,
        prompt: Prompt,
        decoding,
        # decoding: Union[
        #     TopPDecodingStrategy,
        #     TopKDecodingStrategy,
        #     TypicalPDecodingStrategy,
        #     BeamSearchDecodingStrategy,
        # ],
        model_name: str,
        request_id: Optional[str] = None,
        max_retries: int = 2,
        backoff_factor: float = 0.4,
    ):
        if request_id is None:
            request_id = str(uuid.uuid4())

        #print(f"endpoint is {self.__rest_endpoint_generate}")
        request_dict = {"decoding": decoding, "prompt": prompt.dict()}

        for i in range(max_retries):
            try:
                response = self.__session.post(
                    url=self.__rest_endpoint_generate,
                    json=request_dict,
                    params={"model_name": model_name, "request_id": request_id},
                )
                response.raise_for_status()  # Raises a HTTPError if the response contains an HTTP status code that indicates an error
                return response.json()
            except RequestException as e:
                print(f"Request failed with {e}, retrying...{i}")
                temp = prompt.dict()["prompt_template"]
                #print(f"the prompt length was {temp['template'][0]['text']}")
                time.sleep(backoff_factor * (2 ** i))  # Exponential backoff
        #print(f" ################## Error the response is {response.text}  for prompt {prompt.dict()}###############################")
        return None
        

    def tokenize(
        self,
        model_name: str,
        text: str,
        padding: bool = False,
        add_special_tokens: bool = False,
        request_id: Optional[str] = None,
    ) -> List[int]:
        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self.__session.post(
            url=self.__rest_endpoint_tokenize,
            params={
                "text": text,
                "model_name": model_name,
                "request_id": request_id,
                "padding": padding,
                "add_special_tokens": add_special_tokens,
            },
        )

        if response.status_code == 200:
            return response.json()
        elif "Input too long. Max length" in response.text:
            raise InputTextToLong(
                f"For request id {request_id}, large Language Model Service at "
                f"'{self.__rest_endpoint_tokenize}' reported error code "
                f"{response.status_code} and message '{response.text}'."
            )
        else:
            raise ValueError(
                f"Could not make request to Large Language Model Service at '{self.__rest_endpoint_tokenize}', "
                f"see status {response.status_code} and message {response.text} for request id {request_id}"
            )

    def detokenize(
        self, tokens: List[int], model_name: str, request_id: Optional[str] = None
    ) -> str:
        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self.__session.post(
            url=self.__rest_endpoint_detokenize,
            json=tokens,
            params={"model_name": model_name, "request_id": request_id},
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(
                f"Could not make request to Large Language Model Service at '{self.__rest_endpoint_detokenize}', "
                f"see status {response.status_code} and message {response.text} for request id {request_id}"
            )

    def available_models(self, request_id: Optional[str] = None) -> Dict[str, List]:
        if request_id is None:
            request_id = str(uuid.uuid4())

        response = self.__session.get(url=self.__rest_endpoint_available_models)

        if response.status_code == 200:
            return response.json()
        else:
            raise ValueError(
                f"Could not make request to Large Language Model Service at '{self.__rest_endpoint_available_models}', "
                f"see status {response.status_code} and message {response.text} for request id {request_id}"
            )
