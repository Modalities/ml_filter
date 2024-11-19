import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from ml_filter.data_processing.document_processor import DocumentProcessor
from ml_filter.data_processing.llm_score_metrics import EducationalScoreMetric
from ml_filter.data_processing.prompt_builder import PromptBuilder
from ml_filter.llm_api.llm_rest_client import LLMRestClient
from ml_filter.tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer


def test_run(tmpdir: Path):
    llm_rest_client = Mock(spec=LLMRestClient)
    expected_score = 5

    llm_rest_client.generate = lambda prompt: {
        "generated_text": f"{p['content']} score:{expected_score}" for p in prompt
    }

    raw_data_path = tmpdir / "raw_data.jsonl"
    with open(raw_data_path, "w") as f:
        json.dump({"text": "some text"}, f)
        json.dump({"text": "some more text"}, f)

    llm_rest_client.tokenizer = Mock(spec=PreTrainedHFTokenizer)
    llm_rest_client.tokenizer.truncation = False
    llm_rest_client.tokenizer.padding = False
    llm_rest_client.tokenizer.max_length = 100

    prompt_builder = Mock(spec=PromptBuilder)
    prompt_builder.construct_prompt = lambda text: [{"role": "user", "content": text}]

    document_processor = DocumentProcessor(
        llm_rest_client=llm_rest_client,
        prompt_builder=prompt_builder,
        queue_size=2,
        batch_size=2,
        raw_data_file_path=raw_data_path,
        experiment_dir_path=tmpdir / "experiment",
        num_processes=1,
        score_metric_name="educational_score",
    )

    try:
        document_processor.run()
    except Exception:
        pytest.fail("document_processor.run() raised an exception unexpectedly")


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
