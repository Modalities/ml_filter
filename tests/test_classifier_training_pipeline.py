import os
from pathlib import Path

import pytest
import torch

from ml_filter.classifier_training_pipeline import ClassifierTrainingPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
working_dir = Path(os.path.dirname(__file__))


@pytest.fixture
def classifier_training_pipeline():
    return ClassifierTrainingPipeline(working_dir / "resources" / "configs" / "test_config.yaml")


@pytest.fixture
def classifier_training_pipeline_multiscore():
    return ClassifierTrainingPipeline(working_dir / "resources" / "configs" / "test_config_multiscore.yaml")


def test_train_classifier(classifier_training_pipeline):
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


def test_train_classifier_multiscore(classifier_training_pipeline_multiscore):
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
