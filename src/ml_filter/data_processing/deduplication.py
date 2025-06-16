import json
import hashlib
from pathlib import Path
from typing import TextIO


def deduplicate_jsonl(
    input_file_path: Path,
    output_file_path: Path,
    batch_size: int = 10000
    ) -> None:
    """
    Deduplicates entries in a JSONL file based on 'doc_id' and 'text' fields.

    Args:
        input_file_path (Path): Path to the input JSONL file.
        output_file_path (Path): Path to the output JSONL file with deduplicated entries.
        batch_size (int): Number of entries to process in each batch. Default is 10000.

    Returns:
        None
    """
    seen_doc_ids = set()
    seen_text_hashes = set()
    deduplicated_entries = []
    print(f"Processing file {input_file_path}")
    with input_file_path.open("r") as f_in:
        with output_file_path.open("w") as f_out:
            for line in f_in:
                if line == "\n":
                    continue
                try:
                    entry = json.loads(line)
                except json.decoder.JSONDecodeError:
                    print(f"{line=}")
                    raise
                # Hash the 'text' field for efficient comparison
                text_hash = hashlib.sha256(entry.get("text", "").encode("utf-8")).hexdigest()
                doc_id = entry.get("id")
                if doc_id not in seen_doc_ids:
                    seen_doc_ids.add(doc_id)
                    if text_hash not in seen_text_hashes:
                        seen_text_hashes.add(text_hash)
                        deduplicated_entries.append(entry)

                if len(deduplicated_entries) == batch_size:
                    write_to_output_file(f_out, deduplicated_entries)
                    deduplicated_entries = []
                    
            if deduplicated_entries:
                write_to_output_file(f_out, deduplicated_entries)


def write_to_output_file(f_out: TextIO, entries: list[dict]) -> None:
    """
    Writes a list of JSON-serializable entries to an output file.

    Args:
        f_out (TextIO): The file object to write the entries to.
        entries (list[dict]): A list of dictionary entries to be written to the file.

    Returns:
        None
    """
    for entry in entries:
        f_out.write(json.dumps(entry) + "\n")
