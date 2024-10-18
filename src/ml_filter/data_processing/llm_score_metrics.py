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

    metric_name = "educational_score"
    pattern = r"Educational score:\s*(\d+)"


score_metrics = {"educational_score": EducationalScoreMetric}
