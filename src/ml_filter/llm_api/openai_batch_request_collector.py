import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI
from openai.types.batch import Batch
from requests import Session
from requests.adapters import HTTPAdapter

from ml_filter.data_processing.document import Annotation, DocumentProcessingStatus, ProcessedDocument
from ml_filter.llm_api.llm_rest_client import LLMRestClient
from ml_filter.utils.logging import get_logger


class OpenAIBatchAPIRequestSubmitter:
    def __init__(self, input_files: List[Path], model_name: str, max_requests_per_file: int | None) -> None:
        self.input_files = input_files
        self.model_name = model_name
        self.max_requests_per_file = max_requests_per_file
        self.batch_response_filename = "batch_submission_response.json"
        self.client = OpenAI()
        self.logger = get_logger(name=self.__class__.__name__, level=logging.INFO)

    def check_status_maybe_get_results(self):
        batch_response_filepath = [
            Path(f).parent / self.batch_response_filename for f, _ in self._get_batched_requests().items()
        ]
        for filepath in batch_response_filepath:
            if not filepath.exists():
                raise ValueError(f"Batch response file {filepath} does not exist. First submit the batch request.")
            with open(filepath, "r") as f:
                response_data = json.load(f)
            batch = Batch.model_validate(response_data)
            self.logger.info(f"Checking status for requests file {filepath}:")
            data = self._check_batch(batch)
            if data is not None:
                out_path = filepath.parent / "batch_results.jsonl"
                with open(out_path, "w") as out_f:
                    for item in data:
                        out_f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _check_batch(self, batch: Batch) -> List[Dict[str, Any]] | None:
        batch_check = self.client.batches.retrieve(batch.id)

        status_reason_mapping = {
            "validating": "🔍 The input file is being validated before the batch can begin.",
            "failed": "❌ The input file has failed the validation process.",
            "in_progress": "⚙️ The input file was successfully validated and the batch is currently being run.",
            "finalizing": "⚙️ The batch has completed and the results are being prepared.",
            "completed": "✅ The batch has been completed and the results are ready.",
            "expired": "partial ❌ The batch was not able to be completed within the 24-hour time window.",
            "cancelling": "❌ The batch is being cancelled (may take up to 10 minutes).",
            "cancelled": "❌ The batch was cancelled.",
        }
        if batch_check.status == "failed":
            self.logger.error(
                f"Batch request {batch.id} failed with errors:" + batch_check.errors.model_dump_json(indent=2)
                if batch_check.errors is not None
                else "No error details available."
            )
        else:
            self.logger.info(
                f"Batch request {batch.id} status: {batch_check.status} - {status_reason_mapping[batch_check.status]}"
            )

        if batch_check.status == "completed" and batch_check.request_counts is not None:
            if batch_check.request_counts.completed > 0:
                if batch_check.request_counts.completed == batch_check.request_counts.total:
                    self.logger.info(f"All {batch_check.request_counts.completed} requests completed successfully.")
                else:
                    self.logger.warning(
                        f"{batch_check.request_counts.completed} out "
                        + f"of {batch_check.request_counts.total} requests completed successfully."
                    )
                result = self.client.files.content(batch_check.output_file_id).content.decode("utf-8")
                lines = result.strip().split("\n")
                data = [json.loads(line) for line in lines]
                self.logger.info(result)
                return data
            if batch_check.request_counts.failed > 0:
                self.logger.error(f"{batch_check.request_counts.failed} requests failed:")
                error = self.client.files.content(batch_check.error_file_id).content.decode("utf-8")
                self.logger.error(error)

    def submit(self):
        batch_request_file_paths = self._store_batch_requests()

        for batch_request_file_path in batch_request_file_paths:
            batch_input_file = self.client.files.create(file=open(batch_request_file_path, "rb"), purpose="batch")
            batch_input_file_id = batch_input_file.id
            batch = self.client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
            )

            self.logger.info(f"Submitted batch request {batch.id} for {batch_request_file_path}:")
            self.logger.info(batch.model_dump_json(indent=2))

            batch_submission_response_file_path = batch_request_file_path.parent / self.batch_response_filename
            with open(batch_submission_response_file_path, "w") as f:
                f.write(batch.model_dump_json(indent=2))
            self._check_batch(batch)

    def _store_batch_requests(self) -> List[Path]:
        batch_requests = self._get_batched_requests()
        all_out_filenames = []
        for out_file_path, requests in batch_requests.items():
            out_file_path = Path(out_file_path)
            all_out_filenames.append(out_file_path)
            out_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(out_file_path, "w") as f:
                for request in requests:
                    f.write(json.dumps(request, ensure_ascii=False) + "\n")
            self.logger.info(f"Stored {len(requests)} requests to {out_file_path}")
        return all_out_filenames

    def _get_batched_requests(self) -> Dict[str, List[Dict[str, Any]]]:
        # A single batch may include up to 50,000 requests, and a batch input file can be up to 200 MB in size.
        batch_requests = {}

        removed_generation_kwargs = set()
        for input_file in self.input_files:
            requests = []
            with open(input_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                data = json.loads(line)
                annotation = Annotation(**data)
                request = annotation.request
                if request is None:
                    raise ValueError(
                        "Request is None. "
                        + "Please collect requests first by enabling use_llm_rest_client_request_collector "
                        + "in 'score_documents' entry point."
                    )
                if self.model_name:
                    request["body"]["model"] = self.model_name
                    if "gpt-5" in self.model_name:
                        forbidden_generation_kwargs = ["max_tokens", "temperature", "top_p"]
                        for param in forbidden_generation_kwargs:
                            if param in request["body"]:
                                del request["body"][param]
                                removed_generation_kwargs.add(param)
                else:
                    raise ValueError("OpenAI model name is not set.")
                requests.append(request)
            if self.max_requests_per_file is not None:
                requests = requests[: self.max_requests_per_file]
            # split requests if larger than 50000
            for i in range(0, len(requests), 50000):
                out_file_path = str(
                    input_file.parent.parent / "batched_requests" / input_file.stem / f"offset_{i}.jsonl"
                )
                batch_requests[out_file_path] = requests[i : i + 50000]
        if removed_generation_kwargs:
            self.logger.warning(
                f"Removed generation kwargs from requests: {removed_generation_kwargs}, "
                + f"as the model {self.model_name} does not support it."
            )
        return batch_requests


class LLMRestClientBatchCollector(LLMRestClient):
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
        verbose: bool,
        sampling_params: Dict[str, Any],
    ):
        """Initializes the LLMRestClient."""
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.model_name = model_name
        self.timeout = timeout
        self.max_tokens = max_tokens
        self.verbose = verbose
        self.logger = get_logger(name=self.__class__.__name__, level=logging.INFO)
        self.session = session
        self.sampling_params = sampling_params

        # TODO: Not entirely sure why this is needed now, but it worked fine previously
        self.session.mount("http://", HTTPAdapter(pool_connections=max_pool_connections, pool_maxsize=max_pool_maxsize))

        self.rest_endpoint_generate = (
            f"{rest_endpoint}v1/completions" if rest_endpoint.endswith("/") else f"{rest_endpoint}/v1/completions"
        )

        self.logger.info(f"Using rest endpoint at {self.rest_endpoint_generate}")

    def generate(self, processed_document: ProcessedDocument) -> List[ProcessedDocument]:
        """Generates a response based on the given prompt.
        Args:
            processed_document (ProcessedDocument): The processed document.

        Returns:
            Dict[str, Any]: A dictionary containing the generated response.
        """

        request = dict(
            model="will be overriden",
            messages=processed_document.messages,
            **self.sampling_params,
        )

        processed_document.document_processing_status = DocumentProcessingStatus.REQUEST_COLLECTED
        processed_document.request = {
            "custom_id": processed_document.document_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": request,
        }
        return [processed_document]

    def parse_response(self, response_dict: dict) -> List[str] | None:
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
            return [choice.get("text") for choice in choices]
