
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Optional, List


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


def get_document_scores(path_to_files: list[Path], aggregation: Optional[str], max_score: Optional[int] = None) -> dict[str, dict[str, float]]:
    """
    Extracts the scores and corresponding document ids from a set of jsonl-files. Documents which do not have a score for each annotator are excluded.
    
    Args:
        path_to_files (Tuple[Path]): A tuple of file paths containing annotation scores in JSONL format.
        output_file_path (Path): The output path to save computed metrics as a JSON file.
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
    document_scores = {}

    # Loop through each file
    for file_path in path_to_files:
        # Extract the first part of the filename for labeling (e.g., the version)
        prompt, prompt_lang, model = file_path.stem.split('_')[1:4]
        annotator_id = "_".join([model, prompt, prompt_lang])
        # Read the JSONL file and extract scores for each document
        with open(file_path, 'r') as f:
            for line in f:
                json_obj = json.loads(line)
                orig_scores = json_obj.get('scores')
                
                # there are two variants for missing annotations: -inf and None. We standardize to None here.
                # In addition, convert scores to ints and discard scores larger than max_score
                scores = []
                for score in orig_scores:
                    if score == float("-inf") or score is None:
                        scores.append(None)
                    else:
                        # validate that score is an integer
                        int_score = int(score)
                        if float(score) != int_score:
                            scores.append(None)
                        else:
                            score = int_score
                            # validate that score is larger than max_score
                            if max_score is not None and score > max_score:
                                scores.append(None)
                            else:
                                scores.append(score)
                
                if aggregation is None:
                    # filter out documents with missing annotations
                    if None in scores:
                        continue                       
                    
                doc_id = json_obj.get('document_id')
                
                if not prompt in document_scores:
                    document_scores[prompt] = {}
                
                if doc_id not in document_scores[prompt]:
                    document_scores[prompt][doc_id] = {}
                        
                version = "_".join([prompt_lang, model])     
                if version in document_scores[prompt][doc_id]:
                    raise ValueError(f"Found duplicate score for {annotator_id}")
                
                # count documents with no valid annotations
                if all(score is None for score in scores):
                    document_scores[prompt][doc_id][version] = "invalid"   
                    continue
                        
                # aggregate scores
                if aggregation is None:
                    for i, score in enumerate(scores):
                        document_scores[prompt][doc_id][f"{version}_{i}"] = int(score)
                else:
                    # discard invalid scores for aggregation
                    scores = [score for score in scores if score is not None]
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
                    document_scores[prompt][doc_id][version] = aggr_score
    
    return document_scores
