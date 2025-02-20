
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Optional, List

import pandas as pd


def most_frequent_average(values: List[int]) -> float:
    """
    Finds the most frequent value(s) in a list. If there are ties, returns the average of the tied values.

    Args:
        values (list): A list of values (can be integers, floats, etc.) to analyze.

    Returns:
        float: The most frequent value, or the average of the most frequent values if there is a tie.

    Example:
        >>> values = [4, 2, 2, 3, 3, 5]
        >>> most_frequent_average(values)
        2.5
    """
    
    counts = Counter(values)
    max_frequency = max(counts.values())
    most_frequent_values = [key for key, val in counts.items() if val == max_frequency]
    return sum(most_frequent_values) / len(most_frequent_values)


def get_document_scores(
    path_to_files: list[Path],
    labels: List[float],
    aggregation: Optional[str],
    ) -> dict[str, dict[str, float]]:
    """
    Extracts the scores and corresponding document ids from a set of jsonl-files. Documents which do not have a score for each annotator are excluded.
    
    Args:
        path_to_files (Tuple[Path]): A tuple of file paths containing annotation scores in JSONL format.
        labels (List[float]): A list of possible labels for the annotators.
        aggregation (Optional[str], optional): Specifies how scores for a document from the same file are aggregated.
            Supported values:
            - "mean": Compute the average score.
            - "max": Use the maximum score.
            - "min": Use the minimum score.
            - "majority": Use the score that was voted the most. If there is a tie, take the average of the winners.
            - None: No aggregation (used for individual annotator analysis).

    Raises:
        ValueError: If invalid parameter combinations are provided.

    Returns:
        None
    """
    document_scores = []
        
    # Loop through each file
    for file_path in path_to_files:
        # Extract relevant metadata from the filename
        prompt, prompt_lang, model = file_path.stem.split('_')[1:4]

        # Read the JSONL file and extract scores for each document
        with open(file_path, 'r') as f:
            for line in f:
                json_obj = json.loads(line)

                # replace invalid scores with None
                scores = []
                for score in json_obj.get('scores'):
                    if score is None:
                        scores.append(None)
                    else:
                        score = float(score)
                        if score in labels:
                            scores.append(score)
                        else:
                            scores.append(None)

                # aggregate scores for each document
                scores = [score for score in scores if score is not None]
                if len(scores) == 0:
                    aggr_score = "invalid"
                else:
                    if aggregation == "min":
                        aggr_score = min(scores)
                    elif aggregation == "max":
                        aggr_score = max(scores)
                    elif aggregation == "mean":
                        aggr_score = mean(scores)
                    elif aggregation == "majority":
                        aggr_score = most_frequent_average(scores)
                    else:
                        raise NotImplementedError(f"Aggregation type {aggregation} is not supported.")
                
                document_scores.append({
                    'prompt': prompt,
                    'prompt_lang': prompt_lang,
                    'model': model,
                    'doc_id': json_obj.get('document_id'),
                    'score': aggr_score,
                })
    
    document_scores_df = pd.DataFrame(document_scores)
    return document_scores_df
