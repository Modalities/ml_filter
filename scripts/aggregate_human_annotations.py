
import json
from pathlib import Path
from ml_filter.analysis.aggregate_scores import add_scores_to_documents
from ml_filter.analysis.utils import get_document_scores


def aggregate_human_annotations(
    annotations_file_path: Path,
    output_file_path: Path,
    raw_data_file_path: Path,
    aggregation: str,
    batch_size: int
):
    document_scores_df = get_document_scores(
        path_to_files=[annotations_file_path],
        labels=[0, 1, 2, 3, 4, 5],
        aggregation=aggregation
    )
    document_scores_df["raw_data_file_path"] = str(raw_data_file_path)
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
    with output_file_path.open("r", encoding="utf-8") as f_in:
        new_lines = []
        for line in f_in:
            if line == "\n":
                continue
            json_obj = json.loads(line)
            json_obj.pop("scores", None)
            new_lines.append(json_obj)

    with output_file_path.open("w", encoding="utf-8") as f_out:
        f_out.write("\n".join(json.dumps(obj, ensure_ascii=False) for obj in new_lines))
    
    
if __name__ == "__main__":
    for raw_data_file_path in Path("/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/511_test_documents_educational_content").glob("*.jsonl"):
        aggregate_human_annotations(
            annotations_file_path = Path("/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/experiments/model_size_architecture/annotations/annotations_edu_en_gt.jsonl"),
            output_file_path=Path("/raid/s3/opengptx/user/richard-rutmann/data/eurolingua/511_test_documents_educational_content_aggregated_scores") / raw_data_file_path.name,
            raw_data_file_path=raw_data_file_path,
            aggregation="majority",
            batch_size=100000
        ) 