import os
import multiprocessing
from typing import List

def write_chunk(lines: List[str], output_dir: str, base_filename: str, chunk_number: int) -> None:
    """Writes a list of lines to a JSONL chunk file."""
    chunk_filename = f"{base_filename}_chunk_{chunk_number}.jsonl"
    with open(os.path.join(output_dir, chunk_filename), 'w') as outfile:
        outfile.writelines(lines)

def chunk_jsonl(input_file_path: str, output_dir: str, lines_per_chunk: int = 10000) -> None:
    """
    Chunks a large JSONL file into smaller files with a specified number of lines per chunk.

    Args:
        input_file_path (str): Path to the input JSONL file.
        output_dir (str): Directory where chunk files will be saved.
        lines_per_chunk (int, optional): Number of lines per chunk file. Defaults to 10,000.
    """
    os.makedirs(output_dir, exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(input_file_path))[0]
    chunk_number = 0
    pool = multiprocessing.Pool(processes=os.cpu_count())  # Use all CPU cores

    with open(input_file_path, 'r') as infile:
        lines: List[str] = []

        for line in infile:
            lines.append(line)
            
            # Once we reach the chunk size, write to a new file in parallel
            if len(lines) == lines_per_chunk:
                pool.apply_async(write_chunk, (lines, output_dir, base_filename, chunk_number))
                lines = []  # Reset for next chunk
                chunk_number += 1

        # Handle remaining lines
        if lines:
            pool.apply_async(write_chunk, (lines, output_dir, base_filename, chunk_number))

    pool.close()
    pool.join()  # Wait for all chunks to finish writing
