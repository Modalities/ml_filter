import json
import pytest
from ml_filter.data_processing.deduplication import deduplicate_jsonl, write_to_output_file


@pytest.fixture
def entries():
    entries = [
        {"id": "1", "text": "text1"},
        {"id": "2", "text": "text2"},
        {"id": "1", "text": "text3"}, # Duplicate by id
        {"id": "3", "text": "text2"}, # Duplicate by text
    ]
    return entries


@pytest.fixture
def sample_jsonl_file(tmp_path, entries):
    """Creates a sample JSONL file for testing."""
    file_path = tmp_path / "input.jsonl"
    with file_path.open("w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    return file_path


def test_write_to_output_file(tmp_path, entries):
    """Tests the write_to_output_file function."""
    output_file = tmp_path / "output.jsonl"

    with output_file.open("w") as f_out:
        write_to_output_file(f_out, entries)

    # Verify the output file content
    expected_len = 4
    with output_file.open("r") as f:
        lines = f.readlines()
        assert len(lines) == expected_len
        for i in range(expected_len):
            assert json.loads(lines[i]) == entries[i]


def test_deduplicate_jsonl(sample_jsonl_file, tmp_path):
    """Tests the deduplicate_jsonl function."""
    output_file = tmp_path / "output.jsonl"
    deduplicate_jsonl(sample_jsonl_file, output_file, batch_size=2)

    # Verify the output file content
    with output_file.open("r") as f:
        lines = f.readlines()
        assert len(lines) == 2  # Only unique entries should remain
        deduplicated_data = [json.loads(line) for line in lines]

    # Expected deduplicated data
    expected_data = [
        {"id": "1", "text": "text1"},
        {"id": "2", "text": "text2"},
    ]

    assert deduplicated_data == expected_data
