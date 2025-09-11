from dataclasses import dataclass, field

# General pattern used for extracting the numeric score
VALUE_PATTERN = r"\s*(\d+(?:\.\d+)?)"

@dataclass
class LLMScoreMetric:
    """
    Base class for LLM score metrics.

    Attributes:
        metric_name (str): Name of the metric.
        prefix (str): Regex prefix to identify the score type (e.g., "Educational score:").
        pattern (str): Full regex pattern for extracting the score (auto-generated).
    """
    metric_name: str
    prefix: str
    pattern: str = field(init=False)

    def __post_init__(self):
        self.pattern = rf"{self.prefix}{VALUE_PATTERN}"


@dataclass
class EducationalScoreMetric(LLMScoreMetric):
    def __init__(self):
        super().__init__(metric_name="educational_score", prefix=r"Educational score:")


@dataclass
class AdultScoreMetric(LLMScoreMetric):
    def __init__(self):
        super().__init__(metric_name="adult_score", prefix=r"Adult score:")


@dataclass
class ReasoningScoreMetric(LLMScoreMetric):
    def __init__(self):
        super().__init__(metric_name="reasoning_score", prefix=r"Reasoning score:")


# Factory dictionary to retrieve metric classes by name
score_metrics = {
    "educational_score": EducationalScoreMetric,
    "adult_score": AdultScoreMetric,
    "reasoning_score": ReasoningScoreMetric,
}
