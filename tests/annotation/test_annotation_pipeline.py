import gzip
import json
import os
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from transformers.utils.hub import cached_file
from omegaconf import OmegaConf

from ml_filter.annotation.annotation_pipeline import run_annotation_pipeline  # adjust import if needed


class TestRunAnnotationPipeline(unittest.TestCase):

    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.embeddings_dir = os.path.join(self.tmp_dir, "embeddings")
        self.output_dir = os.path.join(self.tmp_dir, "outputs")
        os.makedirs(self.embeddings_dir, exist_ok=True)

        # Create dummy HDF5 file with 'train' group
        self.h5_file_path = os.path.join(self.embeddings_dir, "dummy_embeddings.h5")
        with h5py.File(self.h5_file_path, "w") as f:
            grp = f.create_group("train")
            grp.create_dataset("embeddings", data=np.random.rand(3, 768).astype(np.float32))
            # Define a UTF-8 string dtype with variable length strings
            dt = h5py.string_dtype(encoding='utf-8')
            doc_ids = np.array([f"testfile_{i}" for i in range(3)], dtype=dt)

            grp.create_dataset("document_id", data=doc_ids)
            grp.attrs["n_samples"] = 3

        # Get the local cached path to the checkpoint
        mistral_ckpt_path = cached_file(
            "Jackal-AI/JQL-Edu-Heads",
            "checkpoints/edu-mistral-snowflake-balanced.ckpt"
        )

        # Create dummy OmegaConf config
        self.config_path = os.path.join(self.tmp_dir, "config.yaml")
        OmegaConf.save(config=OmegaConf.create({
            "embeddings_directory": self.embeddings_dir,
            "regression_head_checkpoints": {
                "Edu-JQL-Mistral-SF": mistral_ckpt_path
            },
            "output_dir": self.output_dir,
            "batch_size": 2,
            "tasks": 1,
            "local_tasks": 1,
            "local_rank_offset": 0,
        }), f=self.config_path)

    def tearDown(self):
        shutil.rmtree(self.tmp_dir)

    def test_config_loading(self):
        cfg = OmegaConf.load(self.config_path)
        self.assertEqual(cfg.embeddings_directory, self.embeddings_dir)
        self.assertEqual(cfg.batch_size, 2)
        self.assertEqual(cfg.output_dir, self.output_dir)

    def test_jql_embedding_reader(self):
        from ml_filter.annotation.datatrove_jql_annotator import JQLEmbeddingReader

        reader = JQLEmbeddingReader(data_folder=self.embeddings_dir)
        docs = list(reader.run())
        self.assertGreater(len(docs), 0)

        doc = docs[0]
        self.assertIsInstance(doc.text, list)  # should be list of floats
        self.assertIn("document_id", doc.metadata)
        self.assertIn("file_path", doc.metadata)

    def test_stats_adapter_format(self):
        from ml_filter.annotation.datatrove_jql_annotator import stats_adapter
        from datatrove.data import Document
        from datatrove.pipeline.writers import JsonlWriter

        dummy_doc = Document(id="123", text="will_be_removed", metadata={"score": 0.95, "label": "positive"})
        dummy_writer = JsonlWriter(output_folder=self.output_dir, output_filename="dummy.jsonl", expand_metadata=True)

        result = stats_adapter(dummy_writer, dummy_doc)

        # Ensure "text" is removed
        self.assertNotIn("text", result)

        # Ensure id is retained
        self.assertEqual(result["id"], "123")

        # Ensure metadata keys are flattened
        self.assertIn("score", result)
        self.assertEqual(result["score"], 0.95)

    def test_run_pipeline(self):
        run_annotation_pipeline(Path(self.config_path))

        # Check that output files exist (handle .jsonl.gz)
        annotated_dir = os.path.join(self.output_dir, "annotated_data")
        files = os.listdir(annotated_dir)
        jsonl_gz_files = [f for f in files if f.endswith(".jsonl.gz")]
        self.assertGreater(len(jsonl_gz_files), 0, "No JSONL.GZ output files found.")

        # Check content of the output
        for file in jsonl_gz_files:
            file_path = os.path.join(annotated_dir, file)
            with gzip.open(file_path, "rt", encoding="utf-8") as f:
                lines = f.readlines()
                self.assertGreater(len(lines), 0, f"No data in output file: {file}")
                for line in lines:
                    obj = json.loads(line)
                    self.assertIn("document_id", obj)
