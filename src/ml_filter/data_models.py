from enum import Enum
from typing import Dict, Union

from pydantic import BaseModel, Field


# Define DecodingStrategy Enum
class DecodingStrategy(str, Enum):
    """Decoding strategies for text generation models"""

    GREEDY = "greedy"
    BEAM_SEARCH = "beam_search"
    TOP_K = "top_k"
    TOP_P = "top_p"


# Base class for decoding strategy parameters
class DecodingParameters(BaseModel):
    """Decoding strategy parameters"""

    strategy: DecodingStrategy


# Decoding strategy parameter classes
class GreedyParameters(DecodingParameters):
    """Greedy decoding strategy parameters"""

    strategy: DecodingStrategy = Field(default=DecodingStrategy.GREEDY)


class BeamSearchParameters(DecodingParameters):
    """Beam search decoding strategy parameters"""

    strategy: DecodingStrategy = Field(default=DecodingStrategy.BEAM_SEARCH)
    num_beams: int = Field(..., gt=0, description="Number of beams must be greater than 0.")
    early_stopping: bool


class TopKParameters(DecodingParameters):
    """Top-K decoding strategy parameters"""

    strategy: DecodingStrategy = Field(default=DecodingStrategy.TOP_K)
    top_k: int = Field(..., gt=0, description="Number of top candidates to consider. Must be greater than 0.")
    temperature: float = Field(..., gt=0, description="Sampling temperature. Must be greater than 0.")


class TopPParameters(DecodingParameters):
    """Top-P decoding strategy parameters"""

    strategy: DecodingStrategy = Field(default=DecodingStrategy.TOP_P)
    top_p: float = Field(
        ..., gt=0, le=1, description="Cumulative probability for nucleus sampling. Must be in the range (0, 1]."
    )
    temperature: float = Field(..., gt=0, description="Sampling temperature. Must be greater than 0.")


# General Information about a document
class DocumentInfo(BaseModel):
    """General information about a document"""

    document_id: str
    prompt: str
    prompt_lang: str
    raw_data_path: str
    model: str
    decoding_parameters: Union[GreedyParameters, BeamSearchParameters, TopKParameters, TopPParameters]


class CorrelationMetrics(BaseModel):
    """Correlation metrics for performance evaluation"""

    correlation: Dict[str, Dict[str, float]]  # Correlation per ground truth approach


class TTestResults(BaseModel):
    """T-Test results for performance evaluation"""

    t_test_p_values: Dict[str, float]  # p-values for each ground truth approach


class StatisticReport(BaseModel):
    """Complete statistical report combining various metrics"""

    document_info: DocumentInfo
    correlation_metrics: CorrelationMetrics
    t_test_results: TTestResults
