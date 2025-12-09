from dataclasses import dataclass


@dataclass
class LLMScoreMetric:
    """A class used to represent a scoring metric for a Language Learning Model (LLM).

    Attributes:
        metric_name (str): The name of the metric.
        pattern (str): The pattern used for the metric.
    """

    metric_name: str
    pattern: str


@dataclass
class EducationalScoreMetric(LLMScoreMetric):
    """
    A metric class for extracting educational scores from text.

    This class inherits from `LLMScoreMetric` and is designed to identify and
    process educational scores using a specific regex pattern.

    Attributes:
        metric_name (str): The name of the metric, set to "educational_score".
        pattern (str): The regex pattern used to extract the educational score
            from text. The pattern looks for the phrase "Educational score:"
            followed by one or more digits.
    """

    metric_name: str = "educational_score"
    pattern: str = r"Educational score:\s*(\d+(?:\.\d+)?)"


@dataclass
class AdultScoreMetric(LLMScoreMetric):
    """
    A metric class for extracting educational scores from text.

    This class inherits from `LLMScoreMetric` and is designed to identify and
    process educational scores using a specific regex pattern.

    Attributes:
        metric_name (str): The name of the metric, set to "educational_score".
        pattern (str): The regex pattern used to extract the educational score
            from text. The pattern looks for the phrase "Educational score:"
            followed by one or more digits.
    """

    metric_name: str = "adult_score"
    pattern: str = r"Adult score:\s*(\d+(?:\.\d+)?)"


@dataclass
class PIIScoreMetric(LLMScoreMetric):
    """
    A metric class for extracting PII scores from text.

    This class inherits from `LLMScoreMetric` and is designed to identify and
    process PII scores using a specific regex pattern.

    Attributes:
        metric_name (str): The name of the metric, set to "pii_score".
        pattern (str): The regex pattern used to extract the PII score
            from text. The pattern looks for the phrase "PII score:"
            followed by one or more digits.
    """

    metric_name: str = "pii_score"
    pattern: str = r"PII score:\s*(\d+(?:\.\d+)?)"


score_metrics = {
    "educational_score": EducationalScoreMetric,
    "adult_score": AdultScoreMetric,
    "pii_score": PIIScoreMetric,
}
