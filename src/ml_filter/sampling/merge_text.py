import glob
import json
import os


def merge_texts(dir1: str, dir2: str, dry_run: bool = False):
    """
    For each JSONL file in dir1 matching {language}_sampled_*.jsonl:
      1. Extract the {language} prefix before "_sampled_".
      2. Locate the corresponding file in dir2 with the same {language} prefix.
      3. Load the second file into a dict mapping record IDs to their "text" field.
      4. Iterate through records in the first file:
         - If the record's ID exists in the second file, add the "text" field.
      5. (Unless dry_run) Overwrite the first file with updated records.
    """
    pattern1 = os.path.join(dir1, "*.jsonl")
    for source_uniform_dist in glob.glob(pattern1):
        basename = os.path.basename(source_uniform_dist)
        # Extract language prefix before '_sampled_'
        if "_sampled_" in basename:
            lang = basename.split("_sampled_")[0]
        else:
            print(f"[WARN] File '{basename}' does not match '*_sampled_*.jsonl' pattern, skipping.")
            continue

        # Find matching file in dir2 by language prefix
        pattern2 = os.path.join(dir2, f"{lang}_*.jsonl")
        matches = glob.glob(pattern2)
        if not matches:
            print(f"[WARN] No matching file for language '{lang}' in '{dir2}'")
            continue
        original_fineweb_file = matches[0]
        print(f"Updating '{basename}' with texts from '{os.path.basename(original_fineweb_file)}'")

        # Load texts from the fineweb file by ID
        fineweb_record_id_to_text = {}
        with open(original_fineweb_file, "r", encoding="utf-8") as f2:
            for line in f2:
                try:
                    original_fineweb_record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_id = original_fineweb_record.get("id")
                if rec_id is not None:
                    fineweb_record_id_to_text[rec_id] = original_fineweb_record.get("text", "")

        # Read and update records in the uniform distribution file
        updated_records = []
        with open(source_uniform_dist, "r", encoding="utf-8") as f1:
            for line in f1:
                try:
                    uniform_dist_record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rec_id = uniform_dist_record.get("id")
                if rec_id in fineweb_record_id_to_text:
                    uniform_dist_record["text"] = fineweb_record_id_to_text[rec_id]
                else:
                    raise FileNotFoundError(f"ID '{rec_id}' not found in '{original_fineweb_file}'")
                updated_records.append(uniform_dist_record)

        # Overwrite the uniform distribution file unless dry_run
        if not dry_run:
            with open(source_uniform_dist, "w", encoding="utf-8") as f1:
                for rec in updated_records:
                    f1.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    # Example usage
    input_dir = "/home/abbas-khan/processed_data_natural/validation_set"
    target_dir = "/raid/s3/opengptx/Fineweb_2_500k_both/"
    merge_texts(input_dir, target_dir, dry_run=False)
