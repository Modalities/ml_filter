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
    raw_data_lookup_dir: Path = None,
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
    for f in files:
        # Extract annotator names
        annotator = _extract_annotator_name(f)
        lang = f.parent.name
        
        # Log the tuple of annotator names
        logger.info(f"Aggregate scores in {f}.")
        lang_dir = output_directory / lang
        lang_dir.mkdir(parents=True, exist_ok=True)
        
        document_scores_df = get_document_scores(
            path_to_files=[f],
            aggregation=aggregation,
            labels=labels,
        )
        
        for raw_data_file_path in document_scores_df["raw_data_file_path"].unique():
            document_scores_for_raw_data_df = document_scores_df[document_scores_df["raw_data_file_path"] == raw_data_file_path]
            duplicated = document_scores_for_raw_data_df["doc_id"].duplicated()
            if duplicated.any():
                duplicate_doc_ids = document_scores_for_raw_data_df.loc[
                    duplicated, "doc_id"
                ].tolist()
                logger.warning(f"Found duplicates in {raw_data_file_path}: {duplicate_doc_ids}")
            if raw_data_lookup_dir is not None:
                raw_data_file_path = raw_data_lookup_dir / Path(raw_data_file_path).name
            else:
                raw_data_file_path = Path(raw_data_file_path)
            output_file_path = output_directory / lang_dir / (raw_data_file_path.stem + f"_{annotator}_aggregated_scores_{aggregation}.jsonl")
            document_scores_for_raw_data_dict = document_scores_for_raw_data_df.set_index("doc_id")["score"].to_dict()

            add_scores_to_documents(
                output_file_path=output_file_path,
                raw_data_file_path=raw_data_file_path,
                document_scores_for_raw_data_dict=document_scores_for_raw_data_dict,
                aggregation=aggregation,
                batch_size=batch_size,
                id_field="id",
            )                             
                    

def add_scores_to_documents(
    output_file_path: Path,
    raw_data_file_path: Path,
    document_scores_for_raw_data_dict: dict,
    aggregation: str, batch_size: int,
    id_field: str
) -> None:
    """
    Write scores and corresponding documents to a JSONL file.
    
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
    output_file_path.parent.mkdir(exist_ok=True, parents=True)
    try:
        with output_file_path.open("w", encoding="utf-8") as f_out, raw_data_file_path.open("r", encoding="utf-8") as f_in:
            for i, line in enumerate(f_in):
                json_obj = json.loads(line)
                document_id = json_obj[id_field]
                if document_id not in document_scores_for_raw_data_dict:
                    err_msg = f"No scores found for document {document_id}. Skip this file."
                    logger.error(err_msg)
                    raise ValueError(err_msg)
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
        logger.info(f"Aggregated scores added to {output_file_path}.")
    except ValueError:
        logger.error(f"Error processing {raw_data_file_path}.")
        if output_file_path.exists():
            output_file_path.unlink()
    
    
def aggregate_human_annotations(
    annotations_file_path: Path,
    output_file_path: Path,
    raw_data_file_path: Path,
    labels: list[float],
    aggregation: str,
    batch_size: int,
) -> None:
    """
    Aggregate human annotations by comparing them to ground truth data.
    Args:
        annotations_file_path (Path): The path to the annotations file.
        output_file_path (Path): The path to the output file.
        raw_data_file_path (Path): The path to the raw data file.
        labels (list[float]): The list of possible labels.
        aggregation (str): The aggregation method to use for the scores.
        batch_size (int): The number of documents to process in each batch.
    Returns:
        None
    """
    document_scores_df = get_document_scores(
        path_to_files=[annotations_file_path],
        labels=labels,
        aggregation=aggregation
    )
    # The field "raw_data_file_path" is added to the DataFrame to keep track of the original file path
    document_scores_df["raw_data_file_path"] = str(raw_data_file_path)
    
    # Convert the DataFrame to a dictionary for faster lookups and to avoid duplicate entries
    document_scores_dict = document_scores_df.set_index("doc_id")["score"].to_dict()
    add_scores_to_documents(
        output_file_path=output_file_path,
        raw_data_file_path=raw_data_file_path,
        document_scores_for_raw_data_dict=document_scores_dict,
        aggregation=aggregation,
        batch_size=batch_size,
        id_field="document_id",
    )        
    # remove field "scores" from output_file_path
    remove_field_from_jsonl_file(output_file_path, "scores")
    

def remove_field_from_jsonl_file(
    jsonl_file_path: Path,
    field: str
) -> None:
    """
    Remove a field from each JSON object in a JSONL file.
    Args:
        jsonl_file_path (Path): The path to the JSONL file.
        field (str): The field to remove from each JSON object.
    Returns:
        None
    """
    try:
        # Read the JSONL file and remove the specified field from each JSON object
        with jsonl_file_path.open("r", encoding="utf-8") as f_in:
            new_lines = []
            for line in f_in:
                if line == "\n":
                    continue
                json_obj = json.loads(line)
                json_obj.pop(field, None)
                new_lines.append(json_obj)
    except FileNotFoundError:
        logger.error(f"File not found: {jsonl_file_path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON in {jsonl_file_path}: {e}")
        raise
    
    # Write the modified JSON objects back to the JSONL file
    # Using a temporary file to avoid data loss in case of an error during writing
    temp_file_path = jsonl_file_path.with_suffix(".tmp")
    with temp_file_path.open("w", encoding="utf-8") as f_out:
        f_out.write("\n".join(json.dumps(obj, ensure_ascii=False) for obj in new_lines))
    temp_file_path.rename(jsonl_file_path)
