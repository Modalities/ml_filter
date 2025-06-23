import os
import shutil
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np
from datatrove.data import Document
from datatrove.pipeline.base import DocumentsPipeline

from ml_filter.annotation.datatrove_jql_annotator import JQLEmbedder, HDF5Writer
from ml_filter.annotation.embedder import SnowflakeArcticEmbedMV2_0
from ml_filter.data_processing.hash_data_files import compute_file_hash


class JQLEmbedderTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

        # Create one dummy .jsonl file with 3 lines
        jsonl_path = Path(self.tmp_dir) / "doc_0.jsonl"
        lines = [
            "This is document 0.",
            "This is document 1.",
            "This is document 2."
        ]
        with jsonl_path.open("w") as f:
            for line in lines:
                f.write(f"{line}\n")

        file_hash = compute_file_hash(jsonl_path)

        # Create documents using file_path and document_id
        self.input_docs = []
        for i, text in enumerate(lines):
            self.input_docs.append(Document(
                id=str(i),
                text=text,
                metadata={
                    "file_path": f"doc_{i}.jsonl",
                    "document_id": f"{file_hash}_{i}"
                }
            ))

        self.doc_pipeline = DocumentsPipeline(self.input_docs)


class TestJQLEmbedder(JQLEmbedderTestBase):
    def test_embedding_output_structure(self):
        embedder = JQLEmbedder(batch_size=2)
        embedded_docs = list(embedder.run(self.doc_pipeline))

        self.assertEqual(len(embedded_docs), len(self.input_docs))

        for doc in embedded_docs:
            self.assertIn("embedding", doc.metadata)
            embedding = np.array(doc.metadata["embedding"])
            self.assertEqual(embedding.ndim, 1)
            self.assertEqual(embedding.shape[0], 768)

            self.assertIn("document_id", doc.metadata)
            self.assertIsInstance(doc.metadata["document_id"], str)


class TestHDF5Writer(JQLEmbedderTestBase):
    def test_write_and_verify_hdf5_output(self):
        embedder = JQLEmbedder(batch_size=2)
        embedded_docs = list(embedder.run(self.doc_pipeline))

        writer = HDF5Writer(
            output_folder=self.tmp_dir,
            output_filename="output.h5",  # Fixed string, not a template
            dataset_name="train",
            batch_size=10
        )
        for doc in embedded_docs:
            writer.write(doc)
        writer.close()

        h5_path = os.path.join(self.tmp_dir, "000_output.h5")
        with h5py.File(h5_path, "r") as f:
            self.assertIn("train", f)
            group = f["train"]
            self.assertIn("embeddings", group)
            self.assertIn("document_id", group)

            embeddings = group["embeddings"][:]
            doc_ids = group["document_id"][:]

            self.assertEqual(embeddings.shape[0], len(self.input_docs))
            self.assertEqual(embeddings.shape[1], 768)

            for i, doc in enumerate(embedded_docs):
                expected_id = doc.metadata["document_id"]
                actual_id = doc_ids[i].decode("utf-8") if isinstance(doc_ids[i], bytes) else str(doc_ids[i])
                self.assertEqual(actual_id, expected_id)
                np.testing.assert_allclose(embeddings[i], np.array(doc.metadata["embedding"]), rtol=1e-5)


class TestJQLEmbedderMatchesManualEmbedding(JQLEmbedderTestBase):
    def test_jql_embedder_matches_snowflake_embed_class(self):
        device = 'cuda'
        model_wrapper = SnowflakeArcticEmbedMV2_0(device=device)

        # Sample input documents
        texts = [doc.text for doc in self.input_docs]

        # Get manual embeddings using the .embed() method
        manual_embeddings = model_wrapper.embed(texts).cpu().tolist()

        # Run through JQLEmbedder pipeline
        embedder = JQLEmbedder(batch_size=2)
        embedded_docs = list(embedder.run(self.doc_pipeline))

        # Compare each embedding
        for i, doc in enumerate(embedded_docs):
            actual = np.array(doc.metadata["embedding"])
            expected = manual_embeddings[i]
            np.testing.assert_allclose(
                actual,
                expected,
                rtol=1e-5,
                atol=1e-5,
                err_msg=f"Embedding mismatch at index {i}"
            )
