import copy
import logging
import time
import traceback
from http import HTTPStatus
from typing import Any, Dict, List

from requests import RequestException, Session
from requests.adapters import HTTPAdapter
from openai import OpenAI, OpenAIError

from ml_filter.data_processing.document import DocumentProcessingStatus, ProcessedDocument
from ml_filter.utils.logging import get_logger


class LLMRestClient:
    """A class representing a REST client for LLM services, supporting both local TGI and OpenAI API."""

    def __init__(
        self,
        max_retries: int,
        backoff_factor: int,
        model_name: str,
        timeout: int,
        session: Session,
        rest_endpoint: str,
        max_pool_connections: int,
        max_pool_maxsize: int,
        max_tokens: int,
        verbose: bool,
        sampling_params: Dict[str, Any],
        provider: str = "local",
        openai_api_key: str = None,
        openai_base_url: str = None,
    ):
        """Initializes the LLMRestClient."""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.logger = get_logger(name=self.__class__.__name__, level=logging.INFO)
        self.sampling_params = sampling_params
        self.provider = provider.lower()

        if self.provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key is required when provider is 'openai'.")
            self.openai_client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_base_url or "https://api.openai.com/v1",
            )
        else:
            self.session = session
            self.session.mount(
                "http://", HTTPAdapter(pool_connections=max_pool_connections, pool_maxsize=max_pool_maxsize)
            )
            self.rest_endpoint_generate = (
                f"{rest_endpoint}v1/completions" if rest_endpoint.endswith("/") else f"{rest_endpoint}/v1/completions"
            )
            self.logger.info(f"Using rest endpoint at {self.rest_endpoint_generate}")

    def generate(self, processed_document: ProcessedDocument) -> List[ProcessedDocument]:
        """Generates a response based on the given prompt."""
        if self.provider == "openai":
            return self._generate_openai(processed_document)
        else:
            return self._generate_local(processed_document)

    def _generate_openai(self, processed_document: ProcessedDocument) -> List[ProcessedDocument]:
        """Generates a response using the OpenAI API."""
        request = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": processed_document.prompt}],
            "max_tokens": self.sampling_params.get("max_tokens", 500),
            "temperature": self.sampling_params.get("temperature", 0.7),
            "top_p": self.sampling_params.get("top_p", 0.9),
            "n": self.sampling_params.get("n", 1),
        }
        start_time_generation = time.time()
        new_documents = []

        for i in range(self.max_retries):
            try:
                response = self.openai_client.chat.completions.create(**request)
                break
            except OpenAIError as e:
                self.logger.error(f"OpenAI API request failed with {e}, retrying... ({i+1}/{self.max_retries})")
                time.sleep(self.backoff_factor * (2**i))
                if i == self.max_retries - 1:
                    processed_document.document_processing_status = DocumentProcessingStatus.ERROR_SERVER
                    processed_document.errors.append(str(e))
                    self.logger.error(f"OpenAI API request failed after {self.max_retries} retries.")
                    return [processed_document]

        generated_texts = self._parse_openai_response(response)
        for generated_text in generated_texts:
            new_document = copy.deepcopy(processed_document)
            if generated_text is not None:
                new_document.generated_text = generated_text
            else:
                new_document.document_processing_status = DocumentProcessingStatus.ERROR_NO_GENERATED_TEXT
                new_document.errors.append("No generated text in OpenAI response.")
            end_time_generation = time.time()
            time_diff_generation = end_time_generation - start_time_generation
            completion_tokens = response.usage.completion_tokens if hasattr(response.usage, "completion_tokens") else 0
            out_token_per_second = completion_tokens / time_diff_generation if time_diff_generation > 0 else 0
            new_document.out_tokens_per_second = out_token_per_second
            new_document.timestamp = int(end_time_generation)
            new_documents.append(new_document)

        return new_documents

    def _generate_local(self, processed_document: ProcessedDocument) -> List[ProcessedDocument]:
        """Generates a response using the local TGI endpoint."""
        request = {
            "model": self.model_name,
            "prompt": processed_document.prompt,
            **self.sampling_params,
        }
        start_time_generation = time.time()
        for i in range(self.max_retries):
            try:
                response = self.session.post(
                    url=self.rest_endpoint_generate,
                    json=request,
                    timeout=self.timeout,
                )
                break
            except RequestException as e:
                traceback.print_exc()
                print(f"Request failed with {e}, retrying...{i}")
                time.sleep(self.backoff_factor * (2**i))
                if i == self.max_retries - 1:
                    processed_document.document_processing_status = DocumentProcessingStatus.ERROR_SERVER
                    processed_document.errors.append(str(e))
                    print(f"Request failed after {self.max_retries} retries.")
                    return [processed_document]

        new_documents = []
        if response.status_code == HTTPStatus.OK:
            response_dict = response.json()
            generated_texts = self.parse_response(response_dict)
            for generated_text in generated_texts:
                new_document = copy.deepcopy(processed_document)
                if generated_text is not None:
                    new_document.generated_text = generated_text
                else:
                    new_document.document_processing_status = DocumentProcessingStatus.ERROR_NO_GENERATED_TEXT
                    new_document.errors.append(f"Response could not be parsed: {response_dict}")
                end_time_generation = time.time()
                time_diff_generation = end_time_generation - start_time_generation
                out_token_per_second = response_dict["usage"]["completion_tokens"] / time_diff_generation
                new_document.out_tokens_per_second = out_token_per_second
                new_document.timestamp = int(end_time_generation)
                new_documents.append(new_document)
        else:
            processed_document.document_processing_status = DocumentProcessingStatus.ERROR_SERVER
            processed_document.errors.append(f"Request failed with status code {response.status_code}: {response.text}")
            processed_document.timestamp = int(time.time())
            new_documents.append(processed_document)

        return new_documents

    def parse_response(self, response_dict: dict) -> List[str] | None:
        """Parses the response from the local LLM service."""
        choices = response_dict.get("choices")
        if choices is None or len(choices) == 0:
            return None
        return [choice.get("text") for choice in choices]

    def _parse_openai_response(self, response: Any) -> List[str] | None:
        """Parses the response from the OpenAI API."""
        choices = response.choices
        if choices is None or len(choices) == 0:
            return None
        return [choice.message.content for choice in choices]