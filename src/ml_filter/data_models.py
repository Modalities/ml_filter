from enum import Enum
from typing import Dict, Union

from pydantic import BaseModel, Field


# Define DecodingStrategy Enum
class DecodingStrategy(str, Enum):
    greedy = "greedy"
    beam_search = "beam_search"
    top_k = "top_k"
    top_p = "top_p"


# Base class for decoding strategy parameters
class DecodingParameters(BaseModel):
    strategy: DecodingStrategy


# Decoding strategy parameter classes
class GreedyParameters(DecodingParameters):
    strategy: DecodingStrategy = Field(default=DecodingStrategy.greedy)


class BeamSearchParameters(DecodingParameters):
    strategy: DecodingStrategy = Field(default=DecodingStrategy.beam_search)
    num_beams: int
    early_stopping: bool


class TopKParameters(DecodingParameters):
    strategy: DecodingStrategy = Field(default=DecodingStrategy.top_k)
    top_k: int
    temperature: float


class TopPParameters(DecodingParameters):
    strategy: DecodingStrategy = Field(default=DecodingStrategy.top_p)
    top_p: float
    temperature: float


# General Information about a document
class DocumentInfo(BaseModel):
    document_id: str
    prompt: str
    prompt_lang: str
    raw_data_path: str
    model: str
    decoding_parameters: Union[GreedyParameters, BeamSearchParameters, TopKParameters, TopPParameters]


# Statistical correlations for performance evaluation
class CorrelationMetrics(BaseModel):
    correlation: Dict[str, Dict[str, float]]  # Correlation per ground truth approach


# T-Test and p-value results for performance evaluation
class TTestResults(BaseModel):
    t_test_p_values: Dict[str, float]  # p-values for each ground truth approach


# Complete statistical report combining various metrics
class StatisticReport(BaseModel):
    document_info: DocumentInfo
    correlation_metrics: CorrelationMetrics
    t_test_results: TTestResults
