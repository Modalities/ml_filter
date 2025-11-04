"""Tests for score-based filtering pipeline.

Includes:
 - Unit test for `make_filter_func` threshold logic.
 - Parser test validating duplicate handling.
 - Order preservation/content test (basic file existence & size checks).
 - Full round-trip tokenize -> filter -> detokenize verification using real packed data format.

Performance note: A simple tokenizer cache avoids repeatedly reloading the HF tokenizer across helper functions.
"""

# Standard library imports
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from typing import Any

# Third-party imports
import numpy as np
from datatrove.executor import LocalPipelineExecutor

from ml_filter.data_processing.score_based_filtering.step_data_filtering import make_filter_func
from ml_filter.data_processing.score_based_filtering.step_score_parsing import ScoresParser
from ml_filter.data_processing.score_based_filtering.filter_pipeline import build_pipeline

# ---------------------------------------------------------------------------
# Helper constants & functions
# ---------------------------------------------------------------------------
_TOKENIZER_CACHE: dict[str, Any] = {}

HEADER_SIZE = 64  # Mimics EmbeddedStreamData.HEADER_SIZE_IN_BYTES (simplified for tests)
DATA_SECTION_LEN_BYTES = 8
TOKEN_SIZE_DESC_LEN_BYTES = 4


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _get_tokenizer(name: str):
    from transformers import AutoTokenizer
    if name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[name] = AutoTokenizer.from_pretrained(name)
    return _TOKENIZER_CACHE[name]


def _tokenize_to_pbin(raw_jsonl: Path, tokenized_file: Path, tokenizer_name: str = "bert-base-multilingual-cased") -> None:
    """Tokenizer writer producing a valid packed file using modalities header conventions."""
    import pickle
    from modalities.dataloader.create_packed_data import (
        EmbeddedStreamData,
        update_data_length_in_pre_allocated_header,
    )

    tokenizer = _get_tokenizer(tokenizer_name)
    index_list: list[tuple[int, int]] = []
    tokenized_file.parent.mkdir(parents=True, exist_ok=True)
    with raw_jsonl.open("r", encoding="utf-8") as f_in, tokenized_file.open("wb") as f_out:
        # Pre-allocate header with zero data length then patch
        f_out.write((0).to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"))
        f_out.write((4).to_bytes(EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little"))
        header_bytes_written = (
            EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES
            + EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES
        )
        if header_bytes_written < EmbeddedStreamData.HEADER_SIZE_IN_BYTES:
            f_out.write(b"\x00" * (EmbeddedStreamData.HEADER_SIZE_IN_BYTES - header_bytes_written))
        curr_offset = 0
        for line in f_in:
            text = json.loads(line)["text"]
            if not text.strip():
                continue
            enc = tokenizer(text, truncation=True, max_length=None, add_special_tokens=True)
            arr = np.array(enc["input_ids"], dtype=np.uint32)
            bytes_chunk = arr.astype("<u4").tobytes()
            f_out.write(bytes_chunk)
            seg_len = len(bytes_chunk)
            index_list.append((curr_offset, seg_len))
            curr_offset += seg_len
        f_out.write(pickle.dumps(index_list))
    # Patch header with correct data length
    update_data_length_in_pre_allocated_header(tokenized_file, index_list)


def _write_minimal_packed(tokens_per_sample: list[list[int]], path: Path) -> None:
    """Create a minimal packed data file with given tokens per sample using modalities conventions."""
    import pickle
    from modalities.dataloader.create_packed_data import (
        EmbeddedStreamData,
        update_data_length_in_pre_allocated_header,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    index_list: list[tuple[int, int]] = []
    with path.open("wb") as f_out:
        f_out.write((0).to_bytes(EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES, byteorder="little"))
        f_out.write((4).to_bytes(EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES, byteorder="little"))
        header_written = (
            EmbeddedStreamData.DATA_SECTION_LENGTH_IN_BYTES
            + EmbeddedStreamData.TOKEN_SIZE_DESCRIPTOR_LENGTH_IN_BYTES
        )
        if header_written < EmbeddedStreamData.HEADER_SIZE_IN_BYTES:
            f_out.write(b"\x00" * (EmbeddedStreamData.HEADER_SIZE_IN_BYTES - header_written))
        curr_offset = 0
        for sample_tokens in tokens_per_sample:
            arr = np.array(sample_tokens, dtype=np.uint32)
            chunk = arr.astype("<u4").tobytes()
            f_out.write(chunk)
            seg_len = len(chunk)
            index_list.append((curr_offset, seg_len))
            curr_offset += seg_len
        f_out.write(pickle.dumps(index_list))
    update_data_length_in_pre_allocated_header(path, index_list)


def _detokenize_packed(packed_file: Path, tokenizer_name: str = "bert-base-multilingual-cased") -> list[str]:
    """Load a packed .pbin file via modalities and detokenize each sample back to text using the tokenizer.

    This relies on the PackedMemMapDatasetBase interface to iterate samples.
    """
    from modalities.dataloader.dataset import PackedMemMapDatasetBase
    tokenizer = _get_tokenizer(tokenizer_name)
    dataset = PackedMemMapDatasetBase(packed_file, sample_key="input_ids", load_index=True)
    texts: list[str] = []
    for i in range(len(dataset)):
        tokens = dataset[i]["input_ids"].tolist()
        # Handle potential special tokens gracefully
        decoded = tokenizer.decode(tokens, skip_special_tokens=True).strip()
        texts.append(decoded)
    return texts


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
class TestMakeFilterFunc(unittest.TestCase):
    def test_filter_func_threshold_logic(self):
        scores = [
            {"score_A": 1.0, "score_B": 5.0},
            {"score_A": 2.5, "score_B": 4.9},
            {"score_A": 3.0, "score_B": 10.0},
        ]
        thresholds = {"score_A": 2.0, "score_B": 5.0}
        f = make_filter_func(scores, thresholds)
        # Index 0 fails score_A
        self.assertFalse(f((0, {})))
        # Index 1 fails score_B
        self.assertFalse(f((1, {})))
        # Index 2 passes both
        self.assertTrue(f((2, {})))


class TestScoresParserDuplicates(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self.scores_dir = Path(self.tmp_dir) / "scores"
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.tokenized_dir = Path(self.tmp_dir) / "tokenized"
        self.tokenized_dir.mkdir(parents=True, exist_ok=True)
        (self.tokenized_dir / "file1.pbin").write_bytes(b"dummy")
        score_lines = [
            '{"document_id": "file1", "score_A": 1.0, "score_B": 6.0}',
            '{"document_id": "file1", "score_A": 2.0, "score_B": 7.0}',
            '{"document_id": "file2", "score_A": 4.0, "score_B": 9.0}',
        ]
        (self.scores_dir / "file1.jsonl").write_text("\n".join(score_lines) + "\n", encoding="utf-8")
        self.parser = ScoresParser(
            data_folder=str(self.scores_dir),
            score_keys=["score_A", "score_B"],
            tokenized_data_path=self.tokenized_dir,
            base_file_prefix=Path(""),
        )

    def test_parsing_handles_duplicate_ids(self):
        docs_pipeline = self.parser.read_file("file1.jsonl")
        self.assertEqual(len(docs_pipeline), 1)
        doc = docs_pipeline[0]
        metadata = doc.metadata
        score_entries = metadata[ScoresParser.SCORE_ENTRIES_KEY]
        # We expect 3 entries: file1, duplicate renamed to file1_1, and file2
        self.assertEqual(len(score_entries), 3)
        expected_scores = [
            {"score_A": 1.0, "score_B": 6.0},
            {"score_A": 2.0, "score_B": 7.0},
            {"score_A": 4.0, "score_B": 9.0},
        ]
        self.assertEqual(score_entries, expected_scores)


class TestFilteringOrderPreservation(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self.scores_dir = Path(self.tmp_dir) / "scores"
        self.scores_dir.mkdir(parents=True, exist_ok=True)
        self.tokenized_dir = Path(self.tmp_dir) / "tokenized"
        self.tokenized_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = Path(self.tmp_dir) / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # Packed dataset with 5 samples corresponding to order0..order4
        order_file = self.tokenized_dir / "order.pbin"
        order_tokens = [
            [11, 12],
            [13, 14, 15],
            [16],
            [17, 18],
            [19, 20, 21],
        ]
        _write_minimal_packed(order_tokens, order_file)
        self.sample_texts = ["alpha", "bravo", "charlie", "delta", "echo"]
        scores_lines = [
            '{"document_id": "order0", "score_A": 1.0, "score_B": 5.0}',
            '{"document_id": "order1", "score_A": 2.0, "score_B": 5.0}',
            '{"document_id": "order2", "score_A": 2.0, "score_B": 2.0}',
            '{"document_id": "order3", "score_A": 5.0, "score_B": 5.0}',
            '{"document_id": "order4", "score_A": 3.0, "score_B": 7.0}',
        ]
        (self.scores_dir / "order.jsonl").write_text("\n".join(scores_lines) + "\n", encoding="utf-8")
        self.thresholds = {"score_A": 2.0, "score_B": 5.0}
        self.expected_passing_indices = [1, 3, 4]
        # Use real filter_dataset; we'll inspect size not textual content.
    def test_filtered_order_and_content(self):
        pipeline = build_pipeline(
            score_path=self.scores_dir,
            tokenized_data_path=self.tokenized_dir,
            output_folder=self.output_dir,
            thresholds=self.thresholds,
            base_file_prefix=Path(""),
            tokenized_data_extension=".pbin",
        )
        executor = LocalPipelineExecutor(pipeline=pipeline)
        executor.run()
        filtered_files = list(self.output_dir.rglob("order.filtered.pbin"))
        self.assertEqual(len(filtered_files), 1)
        # Basic sanity: filtered file should be non-empty
        self.assertGreater(filtered_files[0].stat().st_size, 0, "Filtered file is unexpectedly empty.")


class TestTokenizeFilterDetokenizeRoundTrip(unittest.TestCase):
    """End-to-end round trip using inlined tokenizer logic and monkeypatched filtering."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)
        self.raw_dir = Path(self.tmp_dir) / "raw"
        self.tokenized_dir = Path(self.tmp_dir) / "tokenized"
        self.output_dir = Path(self.tmp_dir) / "outputs"
        self.scores_dir = Path(self.tmp_dir) / "scores"
        for p in (self.raw_dir, self.tokenized_dir, self.output_dir, self.scores_dir):
            p.mkdir(parents=True, exist_ok=True)

        # Samples and scores definitions
        self.samples = [
            {"text": "hello world"},           # fail A
            {"text": "bonjour le monde"},      # pass
            {"text": "hola mundo"},            # pass
            {"text": "ciao mondo"},            # fail B
        ]
        self.raw_jsonl = self.raw_dir / "sample.jsonl"
        _write_jsonl(self.samples, self.raw_jsonl)
        scores_lines = [
            '{"document_id": "sample0", "score_A": 1.0, "score_B": 6.0}',
            '{"document_id": "sample1", "score_A": 2.0, "score_B": 5.0}',
            '{"document_id": "sample2", "score_A": 3.0, "score_B": 10.0}',
            '{"document_id": "sample3", "score_A": 5.0, "score_B": 2.0}',
        ]
        (self.scores_dir / "sample.jsonl").write_text("\n".join(scores_lines) + "\n", encoding="utf-8")

        # Tokenize
        tokenized_file = self.tokenized_dir / "sample.pbin"
        _tokenize_to_pbin(self.raw_jsonl, tokenized_file)

        # Real filter_dataset will be used; we'll parse output later to approximate pass count via index length.
        self.expected_texts = [self.samples[1]["text"], self.samples[2]["text"]]

    def test_round_trip_filtering(self):
        pipeline = build_pipeline(
            score_path=self.scores_dir,
            tokenized_data_path=self.tokenized_dir,
            output_folder=self.output_dir,
            thresholds={"score_A": 2.0, "score_B": 5.0},
            base_file_prefix=Path(""),
            tokenized_data_extension=".pbin",
        )
        executor = LocalPipelineExecutor(pipeline=pipeline)
        executor.run()
        filtered_files = list(self.output_dir.rglob("sample.filtered.pbin"))
        self.assertEqual(len(filtered_files), 1)
        self.assertGreater(filtered_files[0].stat().st_size, 0)

        # Detokenize filtered file and verify the surviving texts correspond exactly to expected_texts
        detok_texts = _detokenize_packed(filtered_files[0])
        
        # Assert order and content match expected
        self.assertEqual(len(detok_texts), len(self.expected_texts))
        # Simple containment & order check (exact match)
        self.assertEqual(detok_texts, self.expected_texts, "Detokenized filtered texts do not match expected output")


if __name__ == "__main__":
    unittest.main()
