
from collections import Counter
import json
from pathlib import Path
from statistics import mean

import pandas as pd


def most_frequent_average(values: list[int]) -> float:
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
    labels: list[float],
    aggregation: str,
    ) -> dict[str, dict[str, float]]:
    """
    Extracts the scores and corresponding document ids from a set of jsonl-files. Documents which do not have a score for each annotator are excluded.
    
    Args:
        path_to_files (List[Path]): A tuple of file paths containing annotation scores in JSONL format.
        labels (List[float]): A list of possible labels for the annotators.
        aggregation (str, optional): Specifies how scores for a document from the same file are aggregated.
            Supported values:
            - "mean": Compute the average score.
            - "max": Use the maximum score.
            - "min": Use the minimum score.
            - "majority": Use the score that was voted the most. If there is a tie, take the average of the winners.

    Raises:
        ValueError: If invalid parameter combinations are provided.

    Returns:
        None
    """
    document_scores = []
        
    # Loop through each file
    for file_path in path_to_files:
        # Extract relevant metadata from the filename
        prompt, prompt_lang, annotator = file_path.stem.split('__')[1:4]

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
                    'annotator': annotator,
                    'doc_id': json_obj.get('document_id'),
                    'score': aggr_score,
                    'raw_data_file_path': json_obj.get('meta_information', {}).get('raw_data_file_path')
                })
    
    document_scores_df = pd.DataFrame(document_scores)
    return document_scores_df


def round_scores(value: str | int | float) -> str | int:
    """
    Rounds the given value if it is a number, but keeps the value for invalid scores unchanged.

    Args:
        value (Union[str, int, float]): The value to round.

    Returns:
        Union[str, int]: The rounded value or the original value if it is not a number.
    """
    if value == "invalid":
        return value
    return round(value)


def get_common_docs(document_scores_df: pd.DataFrame, annotator_0: str, annotator_1: str) -> pd.DataFrame:
    """
    Gets the common documents annotated by both annotators.

    Args:
        document_scores_df (pd.DataFrame): The DataFrame containing document scores.
        annotator_0 (str): The name of the first annotator.
        annotator_1 (str): The name of the second annotator.

    Returns:
        pd.DataFrame: A DataFrame containing the common documents annotated by both annotators.
    """
    annotator_0_df = document_scores_df[document_scores_df["annotator"] == annotator_0]
    annotator_1_df = document_scores_df[document_scores_df["annotator"] == annotator_1]
    
    # only consider documents that are annotated by both annotators and have valid scores
    common_docs_df = pd.merge(annotator_0_df, annotator_1_df, on=["doc_id", "prompt"], suffixes=("_0", "_1"))
    
    # add rounded scores for each annotator
    for idx in (0, 1):
        common_docs_df[f'rounded_score_{idx}'] = common_docs_df[f'score_{idx}'].apply(round_scores)
    return common_docs_df
