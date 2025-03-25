import json
import logging
from pathlib import Path
from ml_filter.analysis.interrater_reliability import compute_interrater_reliability_metrics
from ml_filter.analysis.utils import get_document_scores
from ml_filter.utils.logging import get_logger

logger = get_logger(name=__name__, level=logging.INFO) # Set up logging


def _extract_annotator_name(filename: Path) -> str:
    """
    Extracts the annotator name from the filename.

    Args:
        filename (Path): The path to the file.

    Returns:
        str: The extracted annotator name.
    """
    basename = filename.stem
    return basename.split("_")[-1]


def aggregate_scores_in_directory(
    input_directory: Path,
    output_directory: Path,
    aggregation: str,
    labels: list[float],
    batch_size: int = 100000,
) -> None:
    """
    Evaluates prompt-based annotations by comparing annotations to ground truth data.

    Args:
        input_directory (Path): The directory containing the annotation files.
        output_directory (Path): The directory to save the evaluation results.
        gt_data (Path): The path to the ground truth data file.
        aggregation (str): The aggregation method to use for the scores.
        labels (list[float]): The list of possible labels.

    Returns:
        None
    """
    # Find all files matching the pattern in the directory and subdirectories
    files = list(input_directory.rglob("annotations_*.jsonl"))

    # Check if there is at least one file
    if len(files) == 0:
        raise ValueError(f"No annotation files found in {input_directory} or its subdirectories.")

    output_directory.mkdir(parents=True, exist_ok=True)

    # Iterate over all pairs of files (tuples)
    for file in files:
        # Extract annotator names
        annotator = _extract_annotator_name(file)
        lang = file.parent.name
        
        # Log the tuple of annotator names
        logger.info(f"Aggregate scores in {file}.")
        lang_dir = output_directory / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        document_scores_df = get_document_scores(
            path_to_files=[file],
            aggregation=aggregation,
            labels=labels,
        )
        
        for raw_data_file_path in document_scores_df["raw_data_file_path"].unique():
            document_scores_for_raw_data_df = document_scores_df[document_scores_df["raw_data_file_path"] == raw_data_file_path]
            if document_scores_for_raw_data_df["doc_id"].duplicated().any():
                raise ValueError("Duplicate doc_id values found in the DataFrame.")
            raw_data_file_path = Path(raw_data_file_path) TODO
            aggr_scores_file_path = output_directory / lang_dir / (raw_data_file_path.stem + f"_{annotator}_aggregated_scores_{aggregation}.jsonl")
            document_scores_for_raw_data_dict = document_scores_for_raw_data_df.set_index("doc_id")["score"].to_dict()

            aggregate_scores(
                aggr_scores_file_path=aggr_scores_file_path,
                raw_data_file_path=raw_data_file_path,
                document_scores_for_raw_data_dict=document_scores_for_raw_data_dict,
                aggregation=aggregation,
                batch_size=batch_size,
            )                        
            logger.info(f"Aggregated scores added to {aggr_scores_file_path} for all entries in {file}.")
                    

def aggregate_scores(aggr_scores_file_path: Path, raw_data_file_path: Path, document_scores_for_raw_data_dict: dict, aggregation: str, batch_size: int) -> None:
    """
    Aggregate scores for a batch of documents and write them to a JSONL file.
    
    Args:
        aggr_scores_file_path (Path): The path to the output file.
        raw_data_file_path (Path): The path to the raw data file.
        document_scores_for_raw_data_dict (dict): A dictionary mapping document IDs to scores.
        aggregation (str): The aggregation method used to compute the scores.
        batch_size (int): The number of documents to process in each batch.     

    Returns:
        None    
    """
    batch = []
    with aggr_scores_file_path.open("w", encoding="utf-8") as f_out, raw_data_file_path.open("r", encoding="utf-8") as f_in:
        for i, line in enumerate(f_in):
            json_obj = json.loads(line)
            document_id = json_obj["id"]
            if document_id not in document_scores_for_raw_data_dict:
                raise ValueError(f"No scores found for document {document_id}.")
            score = document_scores_for_raw_data_dict[document_id]
            json_obj["score"] = score
            json_obj["aggregation_type"] = aggregation
            batch.append(json_obj)
            if len(batch) == batch_size:
                f_out.write("\n".join(json.dumps(obj, ensure_ascii=False) for obj in batch))
                batch = []  # Clear the batch after writing
                logger.info(f"Processed {i+1} documents.")
        if batch:
            f_out.write("\n" + "\n".join(json.dumps(obj, ensure_ascii=False) for obj in batch))
