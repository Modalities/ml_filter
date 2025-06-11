
import csv

from ml_filter.data_processing.hash_data_files import (
    compute_file_hash,
    read_existing_hashes,
    hash_files_to_csv,
)


def create_temp_file_with_content(tmp_path, filename, content):
    file_path = tmp_path / filename
    with open(file_path, "w") as f:
        f.write(content)
    return file_path


def test_compute_file_hash(tmp_path):
    file_path = create_temp_file_with_content(tmp_path, "test.txt", "hello world")
    hash_val = compute_file_hash(file_path)
    # Precomputed MD5 for "hello world"
    assert hash_val == "5eb63bbbe01eeed093cb22bb8f5acdc3"


def test_read_existing_hashes(tmp_path):
    csv_path = tmp_path / "hashes.csv"
    rows = [
        {"file_path": "file1.txt", "md5": "abc"},
        {"file_path": "file2.txt", "md5": "def"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file_path", "md5"])
        writer.writeheader()
        writer.writerows(rows)
    hashes = read_existing_hashes(csv_path)
    assert hashes == {"file1.txt": "abc", "file2.txt": "def"}


def test_hash_files_to_csv_new_file(tmp_path):
    file1 = create_temp_file_with_content(tmp_path, "a.txt", "foo")
    file2 = create_temp_file_with_content(tmp_path, "b.txt", "bar")
    output_csv = tmp_path / "hashes.csv"
    hash_files_to_csv([file1, file2], output_csv, chunk_size=1024)
    with open(output_csv, "r") as f:
        lines = f.read().splitlines()
    assert lines[0] == "file_path,md5"
    assert any("a.txt" in line for line in lines)
    assert any("b.txt" in line for line in lines)


def test_hash_files_to_csv_append(tmp_path):
    # Create initial file and hash
    file1 = create_temp_file_with_content(tmp_path, "a.txt", "foo")
    file2 = create_temp_file_with_content(tmp_path, "b.txt", "bar")
    output_csv = tmp_path / "hashes.csv"
    hash_files_to_csv([file1], output_csv, chunk_size=1024)
    # Now add a new file and call again
    file3 = create_temp_file_with_content(tmp_path, "c.txt", "baz")
    hash_files_to_csv([file1, file2, file3], output_csv, chunk_size=1024)
    with open(output_csv, "r") as f:
        lines = f.read().splitlines()
    # Should contain all three files, but not duplicate file1
    assert sum("a.txt" in line for line in lines) == 1
    assert sum("b.txt" in line for line in lines) == 1
    assert sum("c.txt" in line for line in lines) == 1


def test_hash_files_to_csv_no_new_files(tmp_path, capsys):
    file1 = create_temp_file_with_content(tmp_path, "a.txt", "foo")
    output_csv = tmp_path / "hashes.csv"
    hash_files_to_csv([file1], output_csv, chunk_size=1024)
    # Call again with the same file, should print "No new files to hash."
    hash_files_to_csv([file1], output_csv, chunk_size=1024)
    captured = capsys.readouterr()
    assert "No new files to hash." in captured.out
