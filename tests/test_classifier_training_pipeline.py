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


def test_multiscore_logit_masking(classifier_training_pipeline_multiscore):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dummy_input_ids = torch.tensor([[1, 2, 3], [5, 6, 7]]).to(device)
    batch_size = dummy_input_ids.shape[0]

    classifier_training_pipeline_multiscore.model.to(device)
    output = classifier_training_pipeline_multiscore.model(dummy_input_ids)
    logits = classifier_training_pipeline_multiscore.reshape_and_mask_logits(output["logits"])

    assert logits.shape == (
        batch_size,
        classifier_training_pipeline_multiscore.max_num_labels_per_metric,
        classifier_training_pipeline_multiscore.num_metrics,
    )
    eps = 1e-30
    for i, n_classes in enumerate(classifier_training_pipeline_multiscore.num_classes_per_metric):
        assert (torch.softmax(logits, -1)[:, n_classes:, i] < eps).all()
        assert (torch.softmax(logits, -1)[:, :n_classes, i] > eps).all()


def test_train_classifier(classifier_training_pipeline, classifier_training_pipeline_multiscore):
    try:
        # Act: Call the train method
        classifier_training_pipeline.train_classifier()
        classifier_training_pipeline_multiscore.train_classifier()
    except Exception as e:
        # Fail the test if any exception occurs
        pytest.fail(f"Training raised an unexpected exception: {e}")
