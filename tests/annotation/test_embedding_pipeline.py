import json
import os
import random
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
from omegaconf import OmegaConf

from ml_filter.annotation.embedding_pipeline import run_embedding_pipeline
from ml_filter.data_processing.hash_data_files import hash_files_to_csv, read_existing_hashes


class TestRunEmbeddingPipeline(unittest.TestCase):
    """End-to-end test for the embedding pipeline using a dummy JSONL file."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.tmp_dir, "jsonl_input")
        self.output_dir = os.path.join(self.tmp_dir, "embedding_output")
        os.makedirs(self.input_dir, exist_ok=True)

        # Create 5 JSONL files with varying number of documents (e.g., 2 to 6)
        self.num_files = 5
        self.docs_per_file_list = [random.randint(2, 6) for _ in range(self.num_files)]
        self.total_docs = sum(self.docs_per_file_list)
        self.input_files = []
        self.expected_doc_ids = set()

        for i, num_docs in enumerate(self.docs_per_file_list):
            file_path = os.path.join(self.input_dir, f"input_{i}.jsonl")
            with open(file_path, "w") as f:
                for j in range(num_docs):
                    temp_doc_id = f"temp_{i}_{j}"
                    doc = {
                        "id": temp_doc_id,
                        "text": f"Document text {temp_doc_id}",
                        "metadata": {"document_id": temp_doc_id},
                    }
                    f.write(json.dumps(doc) + "\n")
            self.input_files.append(file_path)

        # Create CSV hashmap for all JSONL files
        self.csv_hashmap_path = Path(self.tmp_dir) / "hashmap.csv"
        hash_files_to_csv([Path(p) for p in self.input_files], self.csv_hashmap_path, chunk_size=1024 * 1024)

        # Read the CSV hashmap to get md5 hashes
        file_hashes = read_existing_hashes(self.csv_hashmap_path)

        # Rewrite JSONL files with hashed doc_ids of form "{md5}_{index}"
        for file_path in self.input_files:
            md5_hash = file_hashes.get(str(file_path))
            if md5_hash is None:
                raise RuntimeError(f"MD5 hash not found for file: {file_path}")

            new_lines = []
            with open(file_path, "r") as f:
                for idx, line in enumerate(f):
                    doc = json.loads(line)
                    hashed_doc_id = f"{md5_hash}_{idx}"
                    doc["id"] = hashed_doc_id
                    doc["metadata"]["document_id"] = hashed_doc_id
                    new_lines.append(json.dumps(doc))
                    self.expected_doc_ids.add(hashed_doc_id)

            with open(file_path, "w") as f:
                f.write("\n".join(new_lines) + "\n")

        # Create OmegaConf config file
        self.config_path = os.path.join(self.tmp_dir, "config.yaml")
        OmegaConf.save(
            config=OmegaConf.create(
                {
                    "dataset_name": "test_dataset",  # optional for interpolation
                    "running_on_slurm": False,
                    "params": {
                        "input_dir": self.input_dir,
                        "output_dir": self.output_dir,
                        "embedding_dir": "embeddings",  # required by builder for output path
                        "csv_hashmap_path": str(self.csv_hashmap_path),
                        "glob_pattern": "*.jsonl",
                        "embedding_model": "Snowflake/snowflake-arctic-embed-m-v2.0",
                        "hdf5_dataset_name": "train",
                        "batch_size": 32,
                        "writer_batch_size": 1000,
                        "max_length": 256,
                        "padding": True,
                        "truncation": True,
                        "save_labels": False,
                    },
                    "local_settings": {
                        "tasks": 2,
                        "workers": -1,
                        "local_tasks": 2,
                        "local_rank_offset": 0,
                    },
                    "slurm_settings": None,
                }
            ),
            f=self.config_path,
        )

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_run_embedding_pipeline_multiple_files(self):
        run_embedding_pipeline(Path(self.config_path))

        embeddings_dir = os.path.join(self.output_dir, "embeddings")
        self.assertTrue(os.path.isdir(embeddings_dir), "Embeddings output directory missing.")

        h5_files = sorted(os.listdir(embeddings_dir))
        self.assertGreaterEqual(len(h5_files), 4, "Expected multiple HDF5 shards due to spillover.")

        all_doc_ids = set()
        total_samples = 0

        for h5_file in h5_files:
            output_file = os.path.join(embeddings_dir, h5_file)
            self.assertTrue(os.path.isfile(output_file), f"HDF5 shard not found: {h5_file}")

            with h5py.File(output_file, "r") as f:
                self.assertIn("train", f, "Missing 'train' group in HDF5 file.")
                grp = f["train"]

                self.assertIn("embeddings", grp)
                self.assertIn("document_id", grp)

                embeddings = grp["embeddings"][:]
                doc_ids = [id.decode() if isinstance(id, bytes) else id for id in grp["document_id"][:]]

                # Check embedding dimension
                self.assertEqual(embeddings.shape[1], 768)

                # Accumulate document IDs and sample counts
                all_doc_ids.update(doc_ids)
                total_samples += embeddings.shape[0]

        self.assertEqual(all_doc_ids, self.expected_doc_ids)
        self.assertEqual(total_samples, len(self.expected_doc_ids))

    def test_embedding_pipeline_spillover(self):
        # Run the pipeline
        run_embedding_pipeline(Path(self.config_path))

        # Check output directory for multiple shards
        embeddings_dir = os.path.join(self.output_dir, "embeddings")
        self.assertTrue(os.path.isdir(embeddings_dir), "Embeddings output directory missing.")

        total_samples = 0
        shard_sample_counts = []
        # Sort input files and hdf5 files to align with shard ordering (000_*.h5 ↔ input_*.jsonl)
        sorted_input_files = sorted(self.input_files)
        h5_files = sorted(Path(embeddings_dir).glob("*.h5"))
        self.assertGreaterEqual(len(h5_files), 4, "Expected multiple HDF5 shards due to spillover.")

        # Zip input jsonl files and corresponding .h5 shards
        for jsonl_path, h5_path in zip(sorted_input_files, h5_files):
            with open(jsonl_path, "r") as jf, h5py.File(h5_path, "r") as hf:
                self.assertIn("train", hf)
                group = hf["train"]

                jsonl_lines = list(jf)
                h5_doc_ids = [d.decode() if isinstance(d, bytes) else d for d in group["document_id"][:]]

                # Count parity
                self.assertEqual(len(jsonl_lines), len(h5_doc_ids),
                                 f"Mismatch in doc count for {jsonl_path} vs {h5_path}")

                # ID alignment
                for i, jsonl_line in enumerate(jsonl_lines):
                    doc = json.loads(jsonl_line)
                    self.assertEqual(doc["metadata"]["document_id"], h5_doc_ids[i])

                embeddings = group["embeddings"][:]
                self.assertEqual(embeddings.shape[0], len(jsonl_lines))
                self.assertEqual(embeddings.shape[1], 768)
                shard_sample_counts.append(embeddings.shape[0])
                total_samples += embeddings.shape[0]

        self.assertEqual(total_samples, self.total_docs,
                         f"Total samples {total_samples} != expected {self.total_docs}")
        self.assertTrue(any(count < total_samples for count in shard_sample_counts),
                        "No spillover detected, all samples in one shard.")
