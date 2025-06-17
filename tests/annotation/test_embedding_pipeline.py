import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
from omegaconf import OmegaConf

from ml_filter.annotation.embedding_pipeline import run_embedding_pipeline  # adjust import path


class TestRunEmbeddingPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.embeddings_dir = os.path.join(self.tmp_dir, "jsonl_input")
        self.output_dir = os.path.join(self.tmp_dir, "embedding_output")
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Create dummy JSONL file with text
        self.sample_docs = [
            {"id": "0", "text": "The sky is blue.", "metadata": {"document_id": "doc_0"}},
            {"id": "1", "text": "The ocean is vast.", "metadata": {"document_id": "doc_1"}},
            {"id": "2", "text": "Mountains are tall.", "metadata": {"document_id": "doc_2"}},
        ]
        jsonl_path = os.path.join(self.embeddings_dir, "sample.jsonl")
        with open(jsonl_path, "w") as f:
            for line in self.sample_docs:
                f.write(json.dumps(line) + "\n")

        # Create OmegaConf config file
        self.config_path = os.path.join(self.tmp_dir, "config.yaml")
        OmegaConf.save(config=OmegaConf.create({
            "embeddings_directory": self.embeddings_dir,
            "output_dir": self.output_dir,
            "tasks": 1,
            "local_tasks": 1,
            "local_rank_offset": 0,
        }), f=self.config_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_run_embedding_pipeline_end_to_end(self):
        run_embedding_pipeline(Path(self.config_path))

        # Verify .h5 output was created
        output_file = os.path.join(
            self.output_dir, "000_sample.h5"
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
            self.assertEqual(set(doc_ids), {"sample_0", "sample_1", "sample_2"})
