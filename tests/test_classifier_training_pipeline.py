import os
from pathlib import Path

import pytest
import torch

from ml_filter.classifier_training_pipeline import ClassifierTrainingPipeline

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
working_dir = Path(os.path.dirname(__file__))


def get_pipeline(relative_config_file_path: str) -> ClassifierTrainingPipeline:
    return ClassifierTrainingPipeline(working_dir / "resources" / "configs" / relative_config_file_path)


def _train_and_test(classifier_training_pipeline: ClassifierTrainingPipeline):
    try:
        classifier_training_pipeline.train_classifier()
    except Exception as e:
        pytest.fail(f"Training raised an unexpected exception: {e}")

    dummy_input_ids = torch.tensor([[1, 2, 3], [5, 6, 7]]).to(classifier_training_pipeline.model.device)
    batch_size = dummy_input_ids.shape[0]

    output = classifier_training_pipeline.model(dummy_input_ids)
    logits = output["logits"]
    return logits, batch_size


def test_train_classifier():
    classifier_training_pipeline = get_pipeline("test_config.yaml")
    logits, batch_size = _train_and_test(classifier_training_pipeline)

    assert logits.shape == (
        batch_size, 
        max(classifier_training_pipeline.num_targets_per_task),
        1
    )
    eps = 1e-30
    for i, n_classes in enumerate(classifier_training_pipeline.num_targets_per_task):
        # nothing is masked
        assert (torch.softmax(logits, dim=-1)[:, :n_classes, i] > eps).all()


def test_train_classifier_multiscore():
    classifier_training_pipeline_multiscore = get_pipeline("test_config_multiscore.yaml")
    logits, batch_size = _train_and_test(classifier_training_pipeline_multiscore)

    assert logits.shape == (
        batch_size,
        max(classifier_training_pipeline_multiscore.num_targets_per_task),
        classifier_training_pipeline_multiscore.num_tasks,
    )
    eps = 1e-30
    for i, n_classes in enumerate(classifier_training_pipeline_multiscore.num_targets_per_task):
        # logits are masked
        assert (torch.softmax(logits, dim=-1)[:, n_classes:, i] < eps).all()
        assert (torch.softmax(logits, dim=-1)[:, :n_classes, i] > eps).all()


def test_train_classifier_regression():
    for config_path in ["test_config_regression.yaml", "test_config_multiscore_regression.yaml"]:
        classifier_training_pipeline_regression = get_pipeline(config_path)
        logits, batch_size = _train_and_test(classifier_training_pipeline_regression)

        assert logits.shape == (
            batch_size, 
            classifier_training_pipeline_regression.num_tasks
        )
        for i, n_classes in enumerate(classifier_training_pipeline_regression.num_targets_per_task):
            # logits are clamped to (0, n_classes - 1)
            assert ((0 <= logits[:, i]) <= n_classes - 1).all()
