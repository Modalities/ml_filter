import json
import logging
import multiprocessing
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from ml_filter.data_processing.document import Annotation, DocumentProcessingStatus, MetaInformation, ProcessedDocument
from ml_filter.data_processing.llm_score_metrics import score_metrics
from ml_filter.data_processing.prompt_builder import PromptBuilder
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
        batch_size: int,
        raw_data_file_paths: List[Path],
        experiment_dir_path: Path,
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
        self.raw_data_file_paths = raw_data_file_paths
        self.experiment_dir_path = experiment_dir_path
        self.strings_to_remove = strings_to_remove

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

    def _process_document(self, document: Dict[str, Any]) -> List[ProcessedDocument]:
        processed_document = ProcessedDocument(
            document_id=document["id"],
            original_text=document["text"],
            raw_data_file_path=document["raw_data_file_path"],
        )
        if "score" in document:
            processed_document.original_score = float(document["score"])
        # text preprocessing
        processed_document.preprocessed_text = self._remove_special_strings(processed_document.original_text)

        # prompt building
        processed_document = self.prompt_builder.construct_prompt(processed_document)

        processed_documents = []
        # text generation
        for _ in range(self.llm_rest_client.num_return_sequences):
            processed_document = self.llm_rest_client.generate(processed_document=processed_document)

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
                logger.warning(f"Error processing document with id {document['id']}: {error_string}")
            processed_documents.append(processed_document)
        return processed_documents

    def _process_documents_batch(self):
        while True:
            batch_of_documents = self.documents_queue.get()

            if batch_of_documents is None:
                break

            annotations = []

            for document in batch_of_documents:
                processed_document_variations = self._process_document(document)
                annotation = self._convert_to_annotation(processed_document_variations)
                annotations.append(annotation)

            self.result_queue.put(annotations)

    def _convert_to_annotation(self, processed_document_variations: List[ProcessedDocument]) -> Annotation:
        annotation = Annotation(
            document_id=processed_document_variations[0].document_id,
            meta_information=MetaInformation(
                prompt=processed_document_variations[0].prompt,
                # TODO
                prompt_lang="en",
                model=self.llm_rest_client.model_name,
                raw_data_file_path=str(processed_document_variations[0].raw_data_file_path),
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
        return len(document["text"]) > 0 and len(document["id"]) > 0

    def _create_batches(self, raw_data_file_paths: List[Path]):
        batch = []
        for raw_data_file_path in raw_data_file_paths:
            with open(raw_data_file_path, "r") as fin:
                while True:
                    document_string = fin.readline()
                    if len(document_string) == 0:
                        break
                    try:
                        document = json.loads(document_string)
                        document["raw_data_file_path"] = raw_data_file_path
                    except json.JSONDecodeError:
                        logger.warning(f"Error decoding document: {document_string}. Skipping.")
                        continue

                    if not self._is_valid_document(document):
                        logger.warning(
                            f"Invalid document with id: {document['id']}. "
                            + "Value of key 'text' has length of 0. Skipping."
                        )
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

    def _write_results(self, experiment_dir_path: Path):
        start_time = time.time()
        results_written = 0
        out_dir_path = experiment_dir_path / "annotations"
        out_dir_path.mkdir(parents=True, exist_ok=True)
        out_file_path = out_dir_path / "processed_documents.jsonl"

        with out_file_path.open("w") as f:
            while True:
                annotations: List[Annotation] = self.result_queue.get()
                if annotations is None:
                    f.flush()
                    break

                for annotation in annotations:
                    json.dump(annotation.model_dump(), f)
                    f.write("\n")
                    results_written += 1

                if results_written % 10 == 0:
                    f.flush()
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    results_per_second = results_written / elapsed_time if elapsed_time > 0 else 0

                    logger.info(
                        f"Results written: {results_written} | Elapsed time: {elapsed_time:.2f} seconds"
                        f" | Results per second: {results_per_second:.2f}"
                    )

        logger.info(
            f"Results written final: {results_written} | Elapsed time: {elapsed_time:.2f} seconds"
            f" | Results per second: {results_per_second:.2f}"
        )

    def run(self):
        """Runs the document processor.

        The documents are split into batches and processed in parallel using multiple processes.
        A set of processes send the pre-procssed documents to the model for scoring,
        while another process writes the results to a file.

        Args:
            documents (Iterable): An iterable containing the documents to be processed.
        """
        reader = multiprocessing.Process(target=self._create_batches, args=(self.raw_data_file_paths,))
        reader.start()

        writer = multiprocessing.Process(target=self._write_results, args=(self.experiment_dir_path,))
        writer.start()

        processor_threads = [
            multiprocessing.Process(target=self._process_documents_batch)
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

        # self._report_statistics()

    def _report_statistics(self):
        report_statistics(output_dir_path=self.experiment_dir_path)


class ReportStats(BaseModel):
    model_config = ConfigDict(extra="allow")
    mae: float
    mse: float
    std: float
    acc: float
    confusion_matrix: Dict


def report_statistics(results_file_path: Path, output_dir_path: Path | None = None) -> Dict[str, Any]:
    """Show the comparison between the original and generated score."""

    annotations = []
    with open(results_file_path, "r") as f:
        for line in f:
            annotations.append(Annotation.model_validate_json(line))
    # TODO use gold_annotation_paths and adapt to Annotation fields
    df = pd.read_json(results_file_path, lines=True)

    df.original_score = df.original_score.astype(float)
    df.score = df.score.astype(float)
    df["score_mae"] = (df["original_score"] - df["score"]).abs().mean()
    df["score_mse"] = ((df["original_score"] - df["score"]) ** 2).mean()
    df["score_std"] = (df["original_score"] - df["score"]).std()
    df["accuracy"] = (df["original_score"] == df["score"]).mean()

    statistics_report = {
        **ReportStats(
            mae=df["score_mae"].mean(),
            mse=df["score_mse"].mean(),
            std=df["score_std"].mean(),
            acc=df["accuracy"].mean(),
            confusion_matrix=pd.crosstab(df["original_score"], df["score"]).to_dict(),
        ).model_dump(),
        "predicted_score_counts": df["score"].value_counts().sort_index().to_dict(),
        "original_score_counts": df["original_score"].value_counts().sort_index().to_dict(),
        "document_status_counts": df["document_processing_status"].value_counts().to_dict(),
        "error_counts": dict(Counter([error for errors in df["errors"].tolist() for error in errors])),
    }
    logger.info(json.dumps(statistics_report, indent=4))

    if output_dir_path is not None:
        with open(output_dir_path / "statistics_report.json", "w") as f:
            json.dump(statistics_report, f, indent=4)

    return statistics_report
