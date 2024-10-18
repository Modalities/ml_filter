import re

from ml_filter.data_processing.llm_score_metrics import EducationalScoreMetric


def test_educational_score_metric():
    educational_score_metric = EducationalScoreMetric()

    assert educational_score_metric.metric_name == "educational_score"
    assert educational_score_metric.pattern is not None

    text = "Hello world! This is a test. score:5"
    assert re.findall(educational_score_metric.pattern, text) == []

    text = "Hello world! This is a test. Educational score:5"
    assert re.findall(educational_score_metric.pattern, text) == ["5"]

    text = "Hello world! This is a test. Educational score:5"
    assert re.findall(educational_score_metric.pattern, text) == ["5"]

    text = "Hello world! This is a test. Educational score:2/5"
    assert re.findall(educational_score_metric.pattern, text) == ["2"]

    text = "Hello world! This is a test. Educational score:2 / 5"
    assert re.findall(educational_score_metric.pattern, text) == ["2"]

    text = "Hello world! This is a test. Educational score: 2 / 5"
    assert re.findall(educational_score_metric.pattern, text) == ["2"]

    text = "Hello world! This is a test. Educational score : 2 / 5"
    assert re.findall(educational_score_metric.pattern, text) == []

    text = "Hello world! This is a test. Educational ScoRe: 5/1"
    assert re.findall(educational_score_metric.pattern, text) == []

    text = "Hello world! Educational score:2/5. This is a test. Educational score: 4/5"
    assert re.findall(educational_score_metric.pattern, text) == ["2", "4"]
