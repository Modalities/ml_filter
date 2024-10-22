import json
import multiprocessing
import os
import re
import sys
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from ml_filter.data_processing.llm_score_metrics import score_metrics
from ml_filter.data_processing.prompt_builder import PromptBuilder
from ml_filter.llm_api.llm_rest_client import LLMRestClient

# Set up logging
logging.basicConfig(level=logging.WARNING)  # Set the logging level as needed
logger = logging.getLogger(__name__)  # Create a logger instance

sys.path.append(os.path.join(os.getcwd(), "src"))


class DocumentProcessor:
    """A class representing a document processor that generates model responses for a given set of documents."""

    def __init__(
        self,
        llm_rest_client: LLMRestClient,
        prompt_builder: PromptBuilder,
        queue_size: int,
        batch_size: int,
        output_file_path: Path,
        num_processes: int,
        score_metric_name: str,
        strings_to_remove: Optional[List[str]] = [],
    ):
        """Initializes the DocumentProcessor."""
        self.llm_rest_client = llm_rest_client
        self.prompt_builder = prompt_builder
        self.documents_queue = multiprocessing.Queue(maxsize=queue_size)
        self.result_queue = multiprocessing.Queue(maxsize=queue_size)
        self.batch_size = batch_size
        self.num_processes = num_processes
        self.output_file_path = output_file_path
        self.strings_to_remove = strings_to_remove

        if score_metric_name not in score_metrics:
            raise ValueError(f"Invalid score metric name: {score_metric_name}.")

        self.score_metric = score_metrics[score_metric_name]

    def find_last_pattern(self, text: str, pattern: str) -> str | None:
        """
        Find the last occurrence of a pattern in the given text.

        Args:
            text (str): The text to search within.
            pattern (str): The regex pattern to search for.
        Returns:
            str | None: The last occurrence of the pattern in the text, or None if no matches are found.
        """

        # Find all occurrences in the text
        matches = re.findall(pattern, text)

        # Return the last occurrence if there are any matches
        return matches[-1][-1] if matches else None

    def _remove_special_strings(self, text: str) -> str:
        """
        Removes specific characters or strings from the input string.

        Args:
            input_string (str): The original string.
            strings_to_remove (list): A list of characters or strings to remove from the input string.

        Returns:
            str: The formatted string with specified characters or strings removed.
        """

        text = text.replace("\n", "").replace("\r", " ")
        for string in list(self.strings_to_remove):
            text = text.replace(string, " ")

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        return text

    def _process_documents_batch(self):
        while True:
            batch_of_documents = self.documents_queue.get()

            if batch_of_documents is None:
                break

            responses = []

            for document in batch_of_documents:
                text = document["text"]
                text = self._remove_special_strings(text)
                prompt = self.prompt_builder.construct_prompt(text)
                error_messages = []
                try:
                    model_response = self.llm_rest_client.generate(prompt=prompt)
                except Exception as e:
                    error_messages.append(f"{type(e)}: {str(e)}")
                    model_response = {}

                if len(model_response) > 0:
                    if "generated_text" not in model_response:
                        error_messages.append(f"Could not find the generated_text in the model_response. Ignore document. Server response: {model_response}")
                    else:
                        score = self.find_last_pattern(model_response["generated_text"], pattern=self.score_metric.pattern)
                        if score is None:
                            error_messages.append("Could not find the score metric '{self.score_metric.metric_name}' in the model response. Ignore document.")
                        else:
                            model_response[self.score_metric.metric_name] = float(score)

                if len(error_messages) > 0:
                    error_string = " | ".join(error_messages)
                    logger.warning(f"Error processing document with id {document['id']}: {error_string}")
                
                model_response["error"] = error_messages
                model_response["id"] = document["id"]
                responses.append(model_response)

            self.result_queue.put(responses)

    def _is_valid_document(self, document: Dict[str, str]) -> bool:
        return len(document["text"]) > 0

    def _create_batches(self, dataset: Iterable):
        batch = []
        for document in tqdm(dataset, desc="Reading documents", disable=True):
            if not self._is_valid_document(document):
                logger.warning(f"Invalid document with id: {document['id']}. Ignoring.")
                continue
            
            batch.append(document)

            if len(batch) % self.batch_size == 0:
                self.documents_queue.put(batch)
                batch = []

        # If there are remaining documents that didn't fill up a batch
        if len(batch) > 0:
            self.documents_queue.put(batch)

        # Add termination signal (None) once all batches are in the queue
        for _ in range(self.num_processes):
            self.documents_queue.put(None)

    def _write_results(self, output_file: str):
        with open(output_file, "w") as f:
            while True:
                results = self.result_queue.get()
                if results is None:
                    break
                for result in results:
                    json.dump(result, f)
                    f.write("\n")

    def run(self, documents: Iterable):
        """Runs the document processor.

        The documents are split into batches and processed in parallel using multiple processes.
        A set of processes send the pre-procssed documents to the model for scoring,
        while another process writes the results to a file.

        Args:
            documents (Iterable): An iterable containing the documents to be processed.
        """
        reader = multiprocessing.Process(target=self._create_batches, args=(documents,))
        reader.start()

        writer = multiprocessing.Process(target=self._write_results, args=(self.output_file_path,))
        writer.start()

        processor_threads = [
            multiprocessing.Process(target=self._process_documents_batch) for _ in range(self.num_processes)
        ]
        for p in processor_threads:
            p.start()
        for p in processor_threads:
            p.join()

        # Stop the writer process.
        # We only need to put once None in the queue because all processor threads already joined
        self.result_queue.put(None)

        writer.join()
