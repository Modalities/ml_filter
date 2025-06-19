import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
from omegaconf import OmegaConf

from ml_filter.annotation.embedding_pipeline import run_embedding_pipeline
from ml_filter.data_processing.hash_data_files import hash_files_to_csv


class TestRunEmbeddingPipeline(unittest.TestCase):
    """End-to-end test for the embedding pipeline using a dummy JSONL file."""

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.tmp_dir, "jsonl_input")
        self.output_dir = os.path.join(self.tmp_dir, "embedding_output")
        os.makedirs(self.input_dir, exist_ok=True)

        # Create dummy JSONL file with text
        self.sample_docs = [
            {"id": "0", "text": "The sky is blue.", "metadata": {"document_id": "doc_0"}},
            {"id": "1", "text": "The ocean is vast.", "metadata": {"document_id": "doc_1"}},
            {"id": "2", "text": "Mountains are tall.", "metadata": {"document_id": "doc_2"}},
        ]
        self.sample_file = os.path.join(self.input_dir, "sample.jsonl")
        with open(self.sample_file, "w") as f:
            for line in self.sample_docs:
                f.write(json.dumps(line) + "\n")

        # Create CSV hashmap for the sample file using your utility
        self.csv_hashmap_path = Path(self.tmp_dir) / "hashmap.csv"
        hash_files_to_csv([Path(self.sample_file)], self.csv_hashmap_path, chunk_size=1024 * 1024)

        # Create OmegaConf config file
        self.config_path = os.path.join(self.tmp_dir, "config.yaml")
        OmegaConf.save(config=OmegaConf.create({
            "input_dir": self.input_dir,
            "output_dir": self.output_dir,
            "tasks": 1,
            "local_tasks": 1,
            "local_rank_offset": 0,
            "csv_hashmap_path": str(self.csv_hashmap_path),
            "glob_pattern": "*.jsonl",
            "embedding_model": 'Snowflake/snowflake-arctic-embed-m-v2.0',
            "hdf5_dataset_name": "train",
        }), f=self.config_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_run_embedding_pipeline_end_to_end(self):
        run_embedding_pipeline(Path(self.config_path))

        # Verify .h5 output was created
        output_file = os.path.join(
            self.output_dir, "embeddings", "000_sample.h5"
        )
        self.assertTrue(os.path.isfile(output_file), "HDF5 file not created.")

        # Check contents of HDF5 file
        with h5py.File(output_file, "r") as f:
            self.assertIn("train", f, "Missing 'train' group in HDF5 file.")
            grp = f["train"]

            self.assertIn("embeddings", grp)
            self.assertIn("document_id", grp)

            embeddings = grp["embeddings"][:]
            doc_ids = [id.decode() if isinstance(id, bytes) else id for id in grp["document_id"][:]]

            self.assertEqual(embeddings.shape[0], len(self.sample_docs))
            self.assertEqual(embeddings.shape[1], 768)  # Check embedding dim
            self.assertEqual(set(doc_ids), {'d437ffe88187d720a372636edfd8dcdf_0', 'd437ffe88187d720a372636edfd8dcdf_1',
                                            'd437ffe88187d720a372636edfd8dcdf_2'})
