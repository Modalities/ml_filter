import json
import logging
import multiprocessing
import os
import re
import time
from pathlib import Path
from queue import Empty
from typing import Any, Dict, List, Optional

import jq
from tqdm import tqdm

from ml_filter.data_processing.document import Annotation, DocumentProcessingStatus, MetaInformation, ProcessedDocument
from ml_filter.data_processing.llm_score_metrics import score_metrics
from ml_filter.data_processing.prompt_builder import PromptBuilder
from ml_filter.data_processing.report_statistics import ThroughputStatistics
from ml_filter.llm_api.llm_rest_client import LLMRestClient

# Set up logging
logging.basicConfig(level=logging.INFO)  # Set the logging level as needed
logger = logging.getLogger(__name__)  # Create a logger instance


class DocumentProcessor:
    """A class representing a document processor that generates model responses for a given set of documents."""

    def __init__(
        self,
        llm_rest_client: LLMRestClient,
        prompt_builder: PromptBuilder,
        queue_size: int,
        raw_data_file_paths: List[Path],
        start_indexes: List[int],
        experiment_dir_path: Path,
        num_processes: int,
        score_metric_name: str,
        jq_language_pattern: str,
        strings_to_remove: Optional[List[str]] = [],
    ):
        """Initializes the DocumentProcessor."""
        self.llm_rest_client = llm_rest_client
        self.prompt_builder = prompt_builder
        self.queue_size = queue_size
        self.documents_queue = multiprocessing.Queue(maxsize=queue_size)
        self.result_queue = multiprocessing.Queue(maxsize=queue_size)
        manager = multiprocessing.Manager()
        self.doc_order = manager.list()  # Use a shared list for document IDs
        self.num_processes = num_processes
        self.raw_data_file_paths = raw_data_file_paths
        # If start_indexes is shorter than raw_data_file_paths, extend it with 0s.
        if len(start_indexes) < len(raw_data_file_paths):
            start_indexes.extend([0] * (len(raw_data_file_paths) - len(start_indexes)))

        self.start_indexes = start_indexes
        self.experiment_dir_path = experiment_dir_path
        self.common_parents_path = Path(os.path.commonpath(raw_data_file_paths))

        self.strings_to_remove = strings_to_remove
        self.jq_language_pattern = jq.compile(jq_language_pattern)

        if score_metric_name not in score_metrics:
            raise ValueError(f"Invalid score metric name: {score_metric_name}.")

        self.score_metric = score_metrics[score_metric_name]

    @staticmethod
    def find_last_pattern(text: str, pattern: str) -> str | None:
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
        return matches[-1] if matches else None

    def _remove_special_strings(self, text: str) -> str:
        """
        Removes specific characters or strings from the input string.

        Args:
            input_string (str): The original string.
            strings_to_remove (list): A list of characters or strings to remove from the input string.

        Returns:
            str: The formatted string with specified characters or strings removed.
        """

        text = text.replace("\n", " ").replace("\r", " ")
        for string in list(self.strings_to_remove):
            text = text.replace(string, " ")

        # Remove extra spaces
        text = re.sub(r"\s+", " ", text)

        return text

    def _process_document(self, document: Dict[str, Any]) -> List[ProcessedDocument]:
        processed_document = ProcessedDocument(
            document_id=document["document_id"],
            original_text=document["text"],
            raw_data_file_path=document["raw_data_file_path"],
            language=document["language"],
        )
        if "score" in document:
            processed_document.original_score = float(document["score"])
        # text preprocessing
        processed_document.preprocessed_text = self._remove_special_strings(processed_document.original_text)

        # prompt building
        processed_document = self.prompt_builder.construct_prompt(processed_document)

        # text generation
        all_processed_documents = []
        processed_documents = self.llm_rest_client.generate(processed_document=processed_document)

        for processed_document in processed_documents:
            # score filtering
            score = DocumentProcessor.find_last_pattern(
                processed_document.generated_text, pattern=self.score_metric.pattern
            )
            if score is None:
                processed_document.document_processing_status = DocumentProcessingStatus.ERROR_FAULTY_SCORE
            else:
                processed_document.score = float(score)
                processed_document.score_type = self.score_metric.metric_name
                processed_document.document_processing_status = DocumentProcessingStatus.SUCCESS

            if len(processed_document.errors) > 0:
                error_string = " | ".join(processed_document.errors)
                logger.warning(f"Error processing document with id {document['document_id']}: {error_string}")
            all_processed_documents.append(processed_document)
        return all_processed_documents

    def _process_documents(self):
        while True:
            document = self.documents_queue.get()

            if document is None:
                break

            processed_document_variations = self._process_document(document)
            annotation = self._convert_to_annotation(processed_document_variations)

            self.result_queue.put(annotation)

    def _convert_to_annotation(self, processed_document_variations: List[ProcessedDocument]) -> Annotation:
        annotation = Annotation(
            document_id=processed_document_variations[0].document_id,
            meta_information=MetaInformation(
                prompt_name=processed_document_variations[0].prompt_name,
                prompt_lang=processed_document_variations[0].language,
                model_name=self.llm_rest_client.model_name,
                raw_data_file_path=str(processed_document_variations[0].raw_data_file_path),
                out_tokens_per_second=processed_document_variations[0].out_tokens_per_second,
            ),
        )
        for processed_document_variation in processed_document_variations:
            annotation.scores.append(processed_document_variation.score)
            annotation.explanations.append(processed_document_variation.generated_text)
            annotation.errors.append(processed_document_variation.errors)
            annotation.time_stamps.append(processed_document_variation.timestamp)
            annotation.document_processing_status.append(processed_document_variation.document_processing_status)

        return annotation

    def _is_valid_document(self, document: Dict[str, str]) -> bool:
        return len(document["text"]) > 0 and len(document["document_id"]) > 0 and len(document["language"]) > 0

    def _load_documents(self, raw_data_file_paths: List[Path], start_indexes: List[int]):
        for raw_data_file_path, index in zip(raw_data_file_paths, start_indexes):
            doc_count = 0
            with open(raw_data_file_path, "r") as fin:
                while True:
                    document_string = fin.readline()
                    if len(document_string) == 0:
                        break

                    # If we haven't reached the start_index yet, skip this document.
                    if doc_count < index:
                        doc_count += 1
                        continue

                    try:
                        document = json.loads(document_string)
                    except json.JSONDecodeError:
                        logger.warning(f"Error decoding document: {document_string}. Skipping.")
                        continue

                    # Add langauge to document based on jq pattern
                    language = self.jq_language_pattern.input(document).first()
                    if not language:
                        raise ValueError(f"Could not find language in document: {document}. Check your jq pattern.")

                    document["language"] = language
                    document["raw_data_file_path"] = raw_data_file_path

                    if not self._is_valid_document(document):
                        logger.warning(
                            f"Invalid document with id: {document['document_id']}. "
                            + "Value of key 'text' has length of 0. Skipping."
                        )
                        continue
                    self.doc_order.append(str(raw_data_file_path) + document["document_id"])
                    self.documents_queue.put(document)

        # Add termination signal (None) once all documents are in the queue
        for _ in range(self.num_processes):
            self.documents_queue.put(None)

    def _write_results(self, common_parents_path: Path):
        """
        Writes processed annotations to output files in the correct order.
        """
        start_time = time.time()
        results_written = 0
        results_dict = {}
        open_files = {}
        total_out_tokens_per_second = 0
        terminate_signal_received = False
        try:
            while True:
                if terminate_signal_received and len(self.doc_order) == 0:
                    break
                try:
                    annotation: Annotation | None = self.result_queue.get(timeout=1)
                    if annotation is None:
                        terminate_signal_received = True
                    else:
                        results_dict[
                            annotation.meta_information.raw_data_file_path + annotation.document_id
                        ] = annotation
                except Empty:
                    pass

                # Get the next document ID in order
                if self.doc_order:
                    next_to_write_id = self.doc_order[0]
                else:
                    next_to_write_id = None

                # Write the next document if available
                if next_to_write_id is not None and next_to_write_id in results_dict:
                    annotation = results_dict.pop(next_to_write_id)
                    self.doc_order.pop(0)

                    # Write the annotation to the output file
                    out_file_path = self._create_out_file_path(annotation, common_parents_path)
                    if out_file_path not in open_files:
                        open_files[out_file_path] = open(out_file_path, "a")  # Append mode

                    f = open_files[out_file_path]
                    json.dump(annotation.model_dump(), f)
                    f.write("\n")
                    results_written += 1
                    total_out_tokens_per_second += annotation.meta_information.out_tokens_per_second

                    # Periodic flush
                    if results_written % 10 == 0:
                        for file in open_files.values():
                            file.flush()
                        elapsed_time = time.time() - start_time
                        results_per_second = results_written / elapsed_time if elapsed_time > 0 else 0
                        logger.info(
                            f"Results written: {results_written} | Elapsed time: {elapsed_time:.2f} seconds "
                            f"| Results per second: {results_per_second:.2f}"
                        )
        finally:
            # Close all open files
            for f in open_files.values():
                f.flush()
                f.close()

        # Final stats logging
        elapsed_time = time.time() - start_time
        results_per_second = results_written / elapsed_time if elapsed_time > 0 else 0
        logger.info(
            f"Final results written: {results_written} | Elapsed time: {elapsed_time:.2f} seconds "
            f"| Results per second: {results_per_second:.2f}"
        )
        with open(self.experiment_dir_path / "throughput.json", "w") as f:
            json.dump(
                ThroughputStatistics(
                    num_documents_written=results_written,
                    elapsed_time_s=elapsed_time,
                    documents_per_second=results_per_second,
                    mean_out_tokens_per_second=total_out_tokens_per_second / results_written,
                    model_name=self.llm_rest_client.model_name,
                    queue_size=self.queue_size,
                    num_processes=self.num_processes,
                    max_new_tokens=self.llm_rest_client.sampling_params["max_tokens"],
                ).model_dump(),
                f,
            )

    def _create_out_file_path(self, annotation: Annotation, common_parents_path: Path) -> Path:
        input_file_path = Path(annotation.meta_information.raw_data_file_path)
        relative_input_parents_path = input_file_path.relative_to(common_parents_path)
        out_file_name = "_".join(
            [
                input_file_path.stem,
                "_annotations",
                annotation.meta_information.model_name.replace("/", "--"),
                annotation.meta_information.prompt_name,
                annotation.meta_information.prompt_lang,
                input_file_path.suffix,
            ]
        )

        out_dir_path = self.experiment_dir_path / "generated_annotations" / relative_input_parents_path.parent
        out_dir_path.mkdir(parents=True, exist_ok=True)
        return out_dir_path / out_file_name

    def run(self):
        """Runs the document processor.

        The documents are split into batches and processed in parallel using multiple processes.
        A set of processes send the pre-procssed documents to the model for scoring,
        while another process writes the results to a file.

        Args:
            documents (Iterable): An iterable containing the documents to be processed.
        """
        reader = multiprocessing.Process(
            target=self._load_documents, args=(self.raw_data_file_paths, self.start_indexes)
        )
        reader.start()

        writer = multiprocessing.Process(target=self._write_results, args=(self.common_parents_path,))
        writer.start()

        processor_threads = [
            multiprocessing.Process(target=self._process_documents)
            for _ in tqdm(range(self.num_processes), desc="Creating processor threads")
        ]
        for p in tqdm(processor_threads, desc="Starting processor threads"):
            p.start()

        for p in processor_threads:
            p.join()

        # Stop the writer process.
        # We only need to put once None in the queue because all processor threads already joined
        self.result_queue.put(None)

        writer.join()
