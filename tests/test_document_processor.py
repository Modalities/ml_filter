from pathlib import Path
from unittest.mock import Mock

import pytest

from ml_filter.data_processing.document_processor import DocumentProcessor
from ml_filter.data_processing.prompt_builder import PromptBuilder
from ml_filter.llm_api.llm_rest_client import LLMRestClient
from ml_filter.tokenizer.tokenizer_wrapper import PreTrainedHFTokenizer


def test_run():
    llm_rest_client = Mock(spec=LLMRestClient)
    expected_score = 5
    HERE = Path(__file__).parent
    out_path = HERE / "cache/predictions.txt"
    llm_rest_client.generate = lambda prompt: {
        "generated_text": f"{p['content']} score:{expected_score}" for p in prompt
    }
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
        output_file_path=Path(out_path),
        num_processes=1,
        score_metric_name="educational_score",
    )

    data = [
        {"text": "Hello world!"},
        {"text": "This is a test."},
    ]

    try:
        document_processor.run(data)
    except Exception:
        pytest.fail("document_processor.run() raised an exception unexpectedly")
