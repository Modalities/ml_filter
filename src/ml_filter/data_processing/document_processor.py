import json
import multiprocessing
import os
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from tqdm import tqdm

from ml_filter.data_processing.prompt_builder import PromptBuilder
from ml_filter.llm_api.llm_rest_client import LLMRestClient

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
                model_response = self.llm_rest_client.generate(prompt=prompt)
                responses.append(model_response["generated_text"])

            self.result_queue.put(responses)

    def _is_valid_document(self, document: Dict[str, str]) -> bool:
        is_valid_document = True
        if len(document["text"]) == 0:
            is_valid_document = False
        if (
            self.llm_rest_client.tokenizer.truncation is False
            and len(document["text"]) > self.llm_rest_client.tokenizer.max_length
        ):
            is_valid_document = False

        return is_valid_document

    def _create_batches(self, dataset: Iterable):
        batch = []
        for document in tqdm(dataset, desc="Reading documents", disable=True):
            if not self._is_valid_document(document):
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
