import hashlib
import csv
from pathlib import Path
from typing import List


def compute_file_hash(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute MD5 hash for a file in chunks (memory efficient).
    
    Args:
        file_path (Path): Path to the file to hash.
        chunk_size (int): Size of chunks to read the file in bytes.
    Returns:
        str: MD5 hash of the file as a hexadecimal string.
    """
    md5 = hashlib.md5()
    with file_path.open("rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def read_existing_hashes(output_csv: Path) -> dict[str, str]:
    """Read existing hashes from the CSV file and return as a dict.
    
    Args:
        output_csv (Path): Path to the CSV file containing hashes.
    Returns:
        dict[str, str]: Dictionary mapping file paths to their MD5 hashes.
    """
    hashes = {}
    if output_csv.exists():
        with output_csv.open("r", newline="") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                hashes[row["file_path"]] = row["md5"]
    return hashes


def hash_files_to_csv(jsonl_files: List[Path], output_csv: Path, chunk_size: int) -> None:
    """
    Computes MD5 hashes for multiple files and writes the hashes
    together with the file paths to a CSV file. If the CSV exists,
    only new files are hashed and appended.
    
    Args:
        jsonl_files (List[Path]): List of file paths to hash.
        output_csv (Path): Path to the output CSV file.
        chunk_size (int): Size of chunks to read files in bytes.
    Returns:
        None
    """
    # Ensure output directory exists
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    # Read existing hashes from the CSV file
    existing_hashes = read_existing_hashes(output_csv)
    
    # Compute hashes for new files
    new_entries = []
    for file_path in jsonl_files:
        file_path_str = str(file_path)
        if file_path_str not in existing_hashes:
            hash_val = compute_file_hash(file_path, chunk_size)
            new_entries.append((file_path_str, hash_val))

    if not new_entries:
        print("No new files to hash.")
        return
    
    # Write new entries to the CSV file
    write_header = not output_csv.exists()
    with output_csv.open("a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if write_header:
            writer.writerow(["file_path", "md5"])
        for entry in new_entries:
            writer.writerow(entry)
            
    print(f"Hashes written to {output_csv}")
