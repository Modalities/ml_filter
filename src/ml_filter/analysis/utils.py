
import json
from pathlib import Path
from statistics import mean
from typing import Union


def get_document_scores(path_to_files: list[Path], aggregation: Union[None, str]) -> dict[str, dict[str, float]]:
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
                
                # filter out documents with missing annotations
                if float("-inf") in json_obj["scores"]:
                    continue
                
                
                doc_id = json_obj.get('document_id')
                
                if not prompt in document_scores:
                    document_scores[prompt] = {}
                
                if doc_id not in document_scores[prompt]:
                    document_scores[prompt][doc_id] = {}
                        
                version = "_".join([prompt_lang, model])                    
                if version in document_scores[prompt][doc_id]:
                    raise ValueError(f"Found duplicate score for {annotator_id}")
                
                # aggregate scores
                scores = json_obj["scores"]
                if aggregation is None:
                    aggr_score = scores
                elif aggregation == "min":
                    aggr_score = min(scores)
                elif aggregation == "max":
                    aggr_score = max(scores)
                elif aggregation == "mean":
                    aggr_score = mean(scores)
                else:
                    raise NotImplementedError(f"Aggregation type {aggregation} is not supported.")
                document_scores[prompt][doc_id][version] = aggr_score
    
    return document_scores
