import copy
import json
from pathlib import Path
from typing import List, Optional
from unittest.mock import Mock

import pandas as pd

from ml_filter.data_processing.document import ProcessedDocument
from ml_filter.data_processing.document_processor import DocumentProcessor
from ml_filter.data_processing.llm_score_metrics import EducationalScoreMetric
from ml_filter.data_processing.prompt_builder import PromptBuilder
from ml_filter.llm_api.llm_rest_client import LLMRestClient
from ml_filter.tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer


def create_mock_document(processed_document: ProcessedDocument, with_errors=False) -> ProcessedDocument:
    expected_score = 5
    new_document = copy.deepcopy(processed_document)
    new_document.generated_text = f"{processed_document.prompt} score:{expected_score}"

    if with_errors and int(processed_document.document_id) > 150:
        new_document.errors = ["Request failed with status code 500"]

    return new_document


def generate_mock(processed_document: ProcessedDocument) -> List[ProcessedDocument]:
    new_document = create_mock_document(processed_document)
    new_document.out_tokens_per_second
    new_document.timestamp
    return [new_document]


def generate_mock_with_errors(processed_document: ProcessedDocument) -> List[ProcessedDocument]:
    new_document = create_mock_document(processed_document, with_errors=True)
    return [new_document]


def construct_prompt_mock(processed_document: ProcessedDocument) -> ProcessedDocument:
    processed_document.prompt = f"My mock prompt: {processed_document.preprocessed_text}"
    processed_document.prompt_name = "mock"
    return processed_document


def create_temp_input_files(tmpdir: Path, num_files: int, num_documents: int) -> List[Path]:
    tmp_input_paths = []
    for k in range(num_files):
        raw_data_path = Path(tmpdir / f"raw_data_{k}.jsonl")
        with open(raw_data_path, "w") as f:
            for i in range(num_documents):
                json.dump({"text": f"some text {i}", "document_id": f"{i}", "language": "en"}, f)
                f.write("\n")
        tmp_input_paths.append(raw_data_path)
    return tmp_input_paths


def initialize_document_processor(
    tmp_input_paths: List[Path], tmpdir: Path, llm_rest_client: LLMRestClient, start_indexes: Optional[List[int]] = []
) -> DocumentProcessor:
    prompt_builder = Mock(spec=PromptBuilder)
    prompt_builder.construct_prompt = construct_prompt_mock

    return DocumentProcessor(
        llm_rest_client=llm_rest_client,
        prompt_builder=prompt_builder,
        queue_size=2,
        raw_data_file_paths=tmp_input_paths,
        experiment_dir_path=Path(tmpdir / "experiment"),
        num_processes=2,
        score_metric_name="educational_score",
        jq_language_pattern=".language",
        start_indexes=start_indexes
    )


def setup_llm_rest_client(generate_mock_func: callable) -> LLMRestClient:
    llm_rest_client = Mock(spec=LLMRestClient)
    llm_rest_client.generate = generate_mock_func

    llm_rest_client.tokenizer = Mock(spec=PreTrainedHFTokenizer)
    llm_rest_client.tokenizer.truncation = False
    llm_rest_client.tokenizer.padding = False
    llm_rest_client.tokenizer.max_length = 100
    llm_rest_client.model_name = "my_model"
    llm_rest_client.sampling_params = {"temperature": 0.5, "max_tokens": 100}

    return llm_rest_client


def test_run(tmpdir: Path):
    llm_rest_client = setup_llm_rest_client(generate_mock)

    tmp_input_paths = create_temp_input_files(tmpdir, num_files=2, num_documents=1000)

    document_processor = initialize_document_processor(tmp_input_paths=tmp_input_paths, tmpdir=tmpdir,
                                                       llm_rest_client=llm_rest_client)

    document_processor.run()

    for results_path in document_processor.common_parents_path.glob("**/*annotations*.jsonl"):
        df = pd.read_json(results_path, lines=True, orient="records")
        assert df["document_id"].tolist() == list(range(len(df))), f"Document ids are not in order in {results_path}"

        assert df.explanations.apply(lambda x: x[0]).tolist() == [
            f"My mock prompt: some text {i} score:5" for i in range(len(df))
        ], "Prompt and text order is not as expected"


def test_find_last_pattern():
    text = "Hello world! This is a test."
    score_metric = EducationalScoreMetric()

    assert DocumentProcessor.find_last_pattern(text, score_metric.pattern) is None

    text = "Hello world! This is a test. score:5"
    assert DocumentProcessor.find_last_pattern(text, score_metric.pattern) is None

    text = "Hello world! This is a test. Educational score:5"
    assert DocumentProcessor.find_last_pattern(text, score_metric.pattern) == "5"

    text = "Hello world! This is a test. Educational score:2/5"
    assert DocumentProcessor.find_last_pattern(text, score_metric.pattern) == "2"

    text = "Hello world! Educational score:2/5. This is a test. Educational score:4/5"
    assert DocumentProcessor.find_last_pattern(text, score_metric.pattern) == "4"

    text = "Hello world! This is a test. Educational ScoRe: 5/1"
    assert DocumentProcessor.find_last_pattern(text, score_metric.pattern) is None


def test_vllm_failure_in_the_middle(tmpdir: Path):
    llm_rest_client = setup_llm_rest_client(generate_mock_with_errors)

    tmp_input_paths = create_temp_input_files(tmpdir, num_files=2, num_documents=1000)

    document_processor = initialize_document_processor(tmp_input_paths=tmp_input_paths, tmpdir=tmpdir,
                                                       llm_rest_client=llm_rest_client)

    document_processor.run()

    # Check if the termination event was set due to the error
    assert document_processor.termination_event.is_set(), "Termination event should be set due to 500 error"

    for idx, results_path in enumerate(document_processor.common_parents_path.glob("**/*annotations*.jsonl")):
        df = pd.read_json(results_path, lines=True, orient="records")
        assert len(
            df) < 150, f"vLLM failure occured. The number of result documents should be smaller than the number of input documents."


def test_annotation_with_start_index(tmpdir: Path):
    llm_rest_client = setup_llm_rest_client(generate_mock)

    tmp_input_paths = create_temp_input_files(tmpdir, num_files=2, num_documents=1000)

    start_indexes = [150]
    document_processor = initialize_document_processor(tmp_input_paths, tmpdir=tmpdir, llm_rest_client=llm_rest_client,
                                                       start_indexes=start_indexes)

    document_processor.run()

    for idx, results_path in enumerate(document_processor.common_parents_path.glob("**/*annotations*.jsonl")):
        df = pd.read_json(results_path, lines=True, orient="records")

        assert df["document_id"].tolist()[0] == start_indexes[idx], "The start index of the annotation should match the start of the document."
