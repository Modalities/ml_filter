
from dataclasses import dataclass


@dataclass
class LLMScoreMetric:     
    metric_name: str 
    pattern: str
    

@dataclass
class EducationalScoreMetric(LLMScoreMetric):
    metric_name = "educational_score"
    pattern = r"Educational score:\s*(\d+)"
    
    
score_metrics = {
    "educational_score": EducationalScoreMetric
}
