import os
from pathlib import Path

import pytest
import torch

from ml_filter.classifier_training_pipeline import ClassifierTrainingPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
working_dir = Path(os.path.dirname(__file__))


def get_pipeline(config_path: str) -> ClassifierTrainingPipeline:
    return ClassifierTrainingPipeline(working_dir / "resources" / "configs" / config_path)


def test_train_classifier():
    classifier_training_pipeline = get_pipeline("test_config.yaml")
    try:
        # Act: Call the train method
        classifier_training_pipeline.train_classifier()
    except Exception as e:
        # Fail the test if any exception occurs
        pytest.fail(f"Training raised an unexpected exception: {e}")

    dummy_input_ids = torch.tensor([[1, 2, 3], [5, 6, 7]]).to(classifier_training_pipeline.model.device)
    batch_size = dummy_input_ids.shape[0]

    output = classifier_training_pipeline.model(dummy_input_ids)
    logits = output["logits"]

    assert logits.shape == (batch_size, int(classifier_training_pipeline.model.num_labels), 1)
    eps = 1e-30
    for i, n_classes in enumerate(classifier_training_pipeline.num_classes_per_metric):
        assert (torch.softmax(logits, dim=-1)[:, :n_classes, i] > eps).all()


def test_train_classifier_multiscore():
    classifier_training_pipeline_multiscore = get_pipeline("test_config_multiscore.yaml")
    try:
        # Act: Call the train method
        classifier_training_pipeline_multiscore.train_classifier()
    except Exception as e:
        # Fail the test if any exception occurs
        pytest.fail(f"Training raised an unexpected exception: {e}")

    dummy_input_ids = torch.tensor([[1, 2, 3], [5, 6, 7]]).to(classifier_training_pipeline_multiscore.model.device)
    batch_size = dummy_input_ids.shape[0]

    output = classifier_training_pipeline_multiscore.model(dummy_input_ids)
    logits = output["logits"]

    assert logits.shape == (
        batch_size,
        max(classifier_training_pipeline_multiscore.num_classes_per_metric),
        classifier_training_pipeline_multiscore.num_metrics,
    )
    eps = 1e-30
    for i, n_classes in enumerate(classifier_training_pipeline_multiscore.num_classes_per_metric):
        assert (torch.softmax(logits, dim=-1)[:, n_classes:, i] < eps).all()
        assert (torch.softmax(logits, dim=-1)[:, :n_classes, i] > eps).all()


def test_train_classifier_regression():
    for config_path in ["test_config_regression.yaml", "test_config_multiscore_regression.yaml"]:
        classifier_training_pipeline_regression = get_pipeline(config_path)
        try:
            # Act: Call the train method
            classifier_training_pipeline_regression.train_classifier()
        except Exception as e:
            # Fail the test if any exception occurs
            pytest.fail(f"Training raised an unexpected exception: {e}")

        dummy_input_ids = torch.tensor([[1, 2, 3], [5, 6, 7]]).to(classifier_training_pipeline_regression.model.device)
        batch_size = dummy_input_ids.shape[0]

        output = classifier_training_pipeline_regression.model(dummy_input_ids)
        logits = output["logits"]

        assert logits.shape == (batch_size, classifier_training_pipeline_regression.num_metrics)
        for i, n_classes in enumerate(classifier_training_pipeline_regression.num_classes_per_metric):
            assert ((0 <= logits[:, i]) <= n_classes - 1).all()
