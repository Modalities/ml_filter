"""Test for LocalExecutionSettings to ensure logging_dir is optional."""

import unittest
from pathlib import Path

from ml_filter.annotation.annotation_pipeline import (
    AnnotationPipelineBuilder,
    AnnotationPipelineParameters,
    LocalExecutionSettings,
)


class TestLocalExecutionSettings(unittest.TestCase):
    """Test LocalExecutionSettings with and without logging_dir."""

    def test_local_execution_settings_without_logging_dir(self):
        """Test that LocalExecutionSettings can be created without logging_dir."""
        local_section = {
            "tasks": 1,
            "local_tasks": 1,
            "local_rank_offset": 0,
            "workers": 4,
        }

        # Should not raise validation error
        settings = LocalExecutionSettings(**local_section)
        self.assertIsNone(settings.logging_dir)
        self.assertEqual(settings.tasks, 1)
        self.assertEqual(settings.workers, 4)

    def test_local_execution_settings_with_logging_dir(self):
        """Test that LocalExecutionSettings works with logging_dir."""
        local_section = {
            "tasks": 2,
            "local_tasks": 2,
            "local_rank_offset": 0,
            "workers": 8,
            "logging_dir": "/tmp/logs",
        }

        settings = LocalExecutionSettings(**local_section)
        self.assertEqual(settings.logging_dir, "/tmp/logs")
        self.assertEqual(settings.tasks, 2)
        self.assertEqual(settings.workers, 8)

    def test_local_execution_settings_all_defaults(self):
        """Test that LocalExecutionSettings can be created with all defaults."""
        settings = LocalExecutionSettings()
        self.assertIsNone(settings.logging_dir)
        self.assertEqual(settings.tasks, 1)
        self.assertEqual(settings.workers, -1)
        self.assertEqual(settings.local_tasks, 1)
        self.assertEqual(settings.local_rank_offset, 0)

    def test_annotation_pipeline_builder_sets_logging_dir(self):
        """Test that AnnotationPipelineBuilder sets logging_dir when it's None."""
        params = AnnotationPipelineParameters(
            embeddings_directory="/tmp/embeddings",
            output_keys=["document_id"],
            output_dir=Path("/tmp/output"),
            regression_head_checkpoints={"model1": "/tmp/ckpt"},
            batch_size=32,
            dataset_name="train",
            compression=None,
            embedding_dtype="float32",
            label_dtype=None,
            model_dtype="float32",
        )

        # Create local_settings without logging_dir
        local_settings = LocalExecutionSettings(tasks=1, workers=1)
        self.assertIsNone(local_settings.logging_dir)

        # Create builder
        builder = AnnotationPipelineBuilder(
            params=params, running_on_slurm=False, local_settings=local_settings
        )

        # After validation, logging_dir should be set
        expected_logging_dir = str(params.output_dir / "logs")
        self.assertEqual(builder.local_settings.logging_dir, expected_logging_dir)


if __name__ == "__main__":
    unittest.main()
