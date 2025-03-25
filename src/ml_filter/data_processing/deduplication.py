import json
import hashlib
from pathlib import Path


def deduplicate_jsonl(input_file: Path, output_file: Path) -> None:
    """
    Deduplicates entries in a JSONL file based on 'doc_id' and 'text' fields.

    Args:
        input_file (Path): Path to the input JSONL file.
        output_file (Path): Path to the output JSONL file with deduplicated entries.

    Returns:
        None
    """
    seen_doc_ids = set()
    seen_text_hashes = set()
    deduplicated_entries = []

    with input_file.open("r") as f_in:
        for line in f_in:
            entry = json.loads(line)
            # Hash the 'text' field for efficient comparison
            text_hash = hashlib.sha256(entry.get("text", "").encode("utf-8")).hexdigest()
            doc_id = entry.get("doc_id")
            if doc_id not in seen_doc_ids:
                seen_doc_ids.add(doc_id)
                if text_hash not in seen_text_hashes:
                    seen_text_hashes.add(text_hash)
                    deduplicated_entries.append(entry)

    with output_file.open("w") as f_out:
        for entry in deduplicated_entries:
            f_out.write(json.dumps(entry) + "\n")

    print(f"Deduplicated file written to {output_file}")
