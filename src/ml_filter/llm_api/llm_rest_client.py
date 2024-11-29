import logging
import time
import traceback
from http import HTTPStatus

from requests import RequestException, Session
from requests.adapters import HTTPAdapter

from ml_filter.data_processing.document import DocumentProcessingStatus, ProcessedDocument


class LLMRestClient:
    """ "A class representing a REST client for the LLM service.
    This class is responsible for sending requests to the LLM service
    (hosted TGI container given the endpoint) and returning the response.
    """

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
        max_new_tokens: int,
        temperature: float,
        verbose: bool,
    ):
        """Initializes the LLMRestClient."""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.verbose = verbose
        self.logger = logging.getLogger(self.__class__.__name__)
        self.session = session

        # TODO: Not entirely sure why this is needed now, but it worked fine previously
        self.session.mount("http://", HTTPAdapter(pool_connections=max_pool_connections, pool_maxsize=max_pool_maxsize))

        self.rest_endpoint_generate = (
            f"{rest_endpoint}v1/completions" if rest_endpoint.endswith("/") else f"{rest_endpoint}/v1/completions"
        )

        self.logger.info(f"Using rest endpoint at {self.rest_endpoint_generate}")

    def generate(self, processed_document: ProcessedDocument) -> ProcessedDocument:
        """Generates a response based on the given prompt.

        Args:
            processed_document (ProcessedDocument): The processed document.

        Returns:
            Dict[str, Any]: A dictionary containing the generated response.
        """

        request = dict(
            model=self.model_name,
            prompt=processed_document.prompt,
            max_tokens=self.max_new_tokens,
            temperature=self.temperature,
        )

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
                    return processed_document

        if response.status_code == HTTPStatus.OK:
            response_dict = response.json()
            generated_text = self.parse_response(response_dict)
            if generated_text is not None:
                processed_document.generated_text = generated_text
            else:
                processed_document.document_processing_status = DocumentProcessingStatus.ERROR_NO_GENERATED_TEXT
                processed_document.errors.append(f"Response could not be parsed: {response_dict}")
        else:
            processed_document.document_processing_status = DocumentProcessingStatus.ERROR_SERVER
            processed_document.errors.append(f"Request failed with status code {response.status_code}: {response.text}")
        return processed_document

    def parse_response(self, response_dict: dict) -> str | None:
        """Parses the response from the LLM service.

        Args:
            response_dict (dict): The response dictionary.

        Returns:
            str: The generated text.
        """
        choices = response_dict.get("choices")
        if choices is None or len(choices) == 0:
            return None
        else:
            return choices[0].get("text")
