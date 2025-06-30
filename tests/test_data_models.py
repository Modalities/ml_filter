import pytest
from pydantic import ValidationError

from ml_filter.data_models import (
    BeamSearchParameters,
    CorrelationMetrics,
    DecodingStrategy,
    DocumentInfo,
    GreedyParameters,
    StatisticReport,
    TopKParameters,
    TopPParameters,
    TTestResults,
)


def test_greedy_parameters():
    params = GreedyParameters()
    assert params.strategy == DecodingStrategy.GREEDY


def test_beam_search_parameters():
    params = BeamSearchParameters(num_beams=10, early_stopping=False)
    assert params.strategy == DecodingStrategy.BEAM_SEARCH
    assert params.num_beams == 10
    assert not params.early_stopping


def test_top_k_parameters():
    params = TopKParameters(top_k=30, temperature=0.7)
    assert params.strategy == DecodingStrategy.TOP_K
    assert params.top_k == 30
    assert params.temperature == 0.7


def test_top_p_parameters():
    params = TopPParameters(top_p=0.85, temperature=0.9)
    assert params.strategy == DecodingStrategy.TOP_P
    assert params.top_p == 0.85
    assert params.temperature == 0.9


def test_invalid_decoding_parameters():
    with pytest.raises(ValidationError):
        BeamSearchParameters(num_beams=-1, early_stopping=False)  # Invalid num_beams
    with pytest.raises(ValidationError):
        TopKParameters(top_k=-5, temperature=0.7)  # Invalid top_k
    with pytest.raises(ValidationError):
        TopPParameters(top_p=1.5, temperature=0.8)  # Invalid top_p


def test_document_info_with_greedy():
    doc_info = DocumentInfo(
        document_id="doc_001",
        prompt="Asses the educational value of the text.",
        prompt_lang="en",
        raw_data_path="/path/to/raw_data.json",
        model="test_model",
        decoding_parameters=GreedyParameters(),
    )
    assert doc_info.document_id == "doc_001"
    assert doc_info.decoding_parameters.strategy == DecodingStrategy.GREEDY


def test_document_info_with_top_p():
    doc_info = DocumentInfo(
        document_id="doc_002",
        prompt="Asses, whether the text contains adult content.",
        prompt_lang="en",
        raw_data_path="/path/to/raw_data.json",
        model="test_model",
        decoding_parameters=TopPParameters(top_p=0.8, temperature=0.6),
    )
    assert doc_info.document_id == "doc_002"
    assert doc_info.decoding_parameters.top_p == 0.8
    assert doc_info.decoding_parameters.temperature == 0.6


def test_statistic_report():
    doc_info = DocumentInfo(
        document_id="doc_003",
        prompt="Asses, whether the text contains chain of thoughts.",
        prompt_lang="en",
        raw_data_path="/path/to/raw_data.json",
        model="test_model",
        decoding_parameters=BeamSearchParameters(num_beams=5, early_stopping=True),
    )
    correlation_metrics = CorrelationMetrics(
        correlation={
            "average": {"pearson": 0.85, "spearman": 0.82},
            "min": {"pearson": 0.75, "spearman": 0.72},
        }
    )
    t_test_results = TTestResults(t_test_p_values={"average": 0.03, "min": 0.05})
    report = StatisticReport(
        document_info=doc_info,
        correlation_metrics=correlation_metrics,
        t_test_results=t_test_results,
    )

    assert report.document_info.document_id == "doc_003"
    assert report.correlation_metrics.correlation["average"]["pearson"] == 0.85
    assert report.t_test_results.t_test_p_values["average"] == 0.03
