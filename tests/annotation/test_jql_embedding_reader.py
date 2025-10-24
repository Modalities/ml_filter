import os
import shutil
import tempfile
import unittest

import h5py
import numpy as np

from ml_filter.annotation.datatrove_jql_annotator import JQLEmbeddingReader


class TestJQLEmbeddingReader(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.tmp_dir = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp_dir)

        # Create a dummy HDF5 file
        self.h5_file = os.path.join(self.tmp_dir, "mock_data.h5")
        self.embeddings = np.random.rand(3, 5).astype(np.float32)
        self.labels = np.array(["doc1", "doc2", "doc3"], dtype=h5py.string_dtype(encoding="utf-8"))

        with h5py.File(self.h5_file, "w") as f:
            grp = f.create_group("train")
            grp.create_dataset("embeddings", data=self.embeddings)
            grp.create_dataset("document_id", data=self.labels)
            grp.attrs["n_samples"] = self.embeddings.shape[0]

    def check_same_data(self, documents, expected_embeddings, expected_document_id, skip=0, limit=None):
        if skip:
            expected_embeddings = expected_embeddings[skip:]
            expected_document_id = expected_document_id[skip:]
        if limit is not None:
            expected_embeddings = expected_embeddings[:limit]
            expected_document_id = expected_document_id[:limit]

        self.assertEqual(len(documents), len(expected_embeddings))

        for i, doc in enumerate(documents):
            np.testing.assert_allclose(doc.text, expected_embeddings[i], rtol=1e-5)
            np.testing.assert_equal(doc.metadata["document_id"], expected_document_id[i])

    def test_read(self):
        reader = JQLEmbeddingReader(self.tmp_dir, "train")
        documents = list(reader.run())
        self.check_same_data(documents, self.embeddings, self.labels)

    def test_read_with_limit(self):
        reader = JQLEmbeddingReader(self.tmp_dir, "train", limit=2)
        documents = list(reader.run())
        self.check_same_data(documents, self.embeddings, self.labels, limit=2)

    def test_read_with_skip(self):
        reader = JQLEmbeddingReader(self.tmp_dir, "train", skip=1)
        documents = list(reader.run())
        self.check_same_data(documents, self.embeddings, self.labels, skip=1)

    def test_read_with_limit_and_skip(self):
        reader = JQLEmbeddingReader(self.tmp_dir, "train", skip=1, limit=1)
        documents = list(reader.run())
        self.check_same_data(documents, self.embeddings, self.labels, skip=1, limit=1)

    def test_handles_missing_dataset_gracefully(self):
        # Create a file with no "train" group
        bad_file = os.path.join(self.tmp_dir, "bad_data.h5")
        with h5py.File(bad_file, "w") as f:
            f.create_group("wrong_group")

        reader = JQLEmbeddingReader(self.tmp_dir, "train", glob_pattern="bad_data.h5")
        documents = list(reader.run())

        # If dataset is missing, documents should be empty
        self.assertEqual(len(documents), 0)
