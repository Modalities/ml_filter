import copy
import json
from pathlib import Path
from typing import List
from unittest.mock import Mock

import pandas as pd

from ml_filter.data_processing.document import ProcessedDocument
from ml_filter.data_processing.document_processor import DocumentProcessor
from ml_filter.data_processing.llm_score_metrics import EducationalScoreMetric
from ml_filter.data_processing.prompt_builder import PromptBuilder
from ml_filter.llm_api.llm_rest_client import LLMRestClient
from ml_filter.tokenization.tokenizer_wrapper import PreTrainedHFTokenizer


def generate_mock(processed_document: ProcessedDocument) -> List[ProcessedDocument]:
    expected_score = 5
    new_document = copy.deepcopy(processed_document)
    new_document.generated_text = f"{processed_document.prompt} score:{expected_score}"
    new_document.out_tokens_per_second
    new_document.timestamp
    return [new_document]


def construct_prompt_mock(processed_document: ProcessedDocument) -> ProcessedDocument:
    processed_document.prompt = f"My mock prompt: {processed_document.preprocessed_text}"
    processed_document.prompt_name = "mock"
    return processed_document


def test_run(tmpdir: Path):
    llm_rest_client = Mock(spec=LLMRestClient)

    llm_rest_client.generate = generate_mock

    tmp_input_paths = []
    for k in range(2):
        raw_data_path = Path(tmpdir / f"raw_data_{k}.jsonl")
        with open(raw_data_path, "w") as f:
            for i in range(1000):
                json.dump({"text": f"some text {i}", "document_id": f"{i}", "language": "en"}, f)
                f.write("\n")
        tmp_input_paths.append(raw_data_path)

    llm_rest_client.tokenizer = Mock(spec=PreTrainedHFTokenizer)
    llm_rest_client.tokenizer.truncation = False
    llm_rest_client.tokenizer.padding = False
    llm_rest_client.tokenizer.max_length = 100
    llm_rest_client.model_name = "my_model"
    llm_rest_client.sampling_params = {"temperature": 0.5, "max_tokens": 100}

    prompt_builder = Mock(spec=PromptBuilder)
    prompt_builder.construct_prompt = construct_prompt_mock

    experiment_dir_path = Path(tmpdir / "experiment")
    document_processor = DocumentProcessor(
        llm_rest_client=llm_rest_client,
        prompt_builder=prompt_builder,
        queue_size=2,
        raw_data_file_paths=tmp_input_paths,
        experiment_dir_path=experiment_dir_path,
        num_processes=2,
        score_metric_name="educational_score",
        jq_language_pattern=".language",
    )

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
