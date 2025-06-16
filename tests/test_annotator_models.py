from typing import Iterable

import pytest
import torch
from transformers import BertForSequenceClassification

from constants import MODEL_CLASS_MAP
from ml_filter.models.annotator_model_head import (
    LogitMaskLayer,
    MultiTargetClassificationHead,
    MultiTargetRegressionHead,
    RegressionScalingLayer,
)
from ml_filter.models.base_model import BaseModel, BaseModelConfig


def test_annotator_model_initialization(annotator_model_config: BaseModelConfig, is_regression: bool):
    """Tests if AnnotatorModel initializes correctly with a base model and head."""
    model = BaseModel(config=annotator_model_config)

    assert model._base_model is not None
    expected_head = MultiTargetRegressionHead if is_regression else MultiTargetClassificationHead
    assert isinstance(model._base_model.classifier, expected_head)


def test_annotator_model_freezing(annotator_model_config: BaseModelConfig):
    """Tests if AnnotatorModel correctly freezes the base model when required."""
    # Create the model with freezing enabled
    model = BaseModel(config=annotator_model_config)
    model.set_freeze_base_model(True)

    # Ensure that not all base model parameters are frozen
    assert not all(
        param.requires_grad for param in model._base_model.parameters()
    ), "All base model parameters are frozen!"

    # Ensure classifier parameters are trainable
    assert all(
        param.requires_grad for param in model._base_model.classifier.parameters()
    ), "Classifier parameters should be trainable!"

    # Special case for BERT: Ensure pooler parameters are trainable
    if hasattr(model._base_model, "bert") and hasattr(model._base_model.bert, "pooler"):
        assert all(
            param.requires_grad for param in model._base_model.bert.pooler.parameters()
        ), "BERT pooler parameters should be trainable!"

    # Ensure all other parameters are frozen
    assert all(
        not param.requires_grad
        for name, param in model._base_model.named_parameters()
        if "classifier" not in name and "pooler" not in name  # Exclude classifier & pooler from check
    ), "Some non-classifier and non-pooler parameters are still trainable!"


def test_annotator_model_unfreezing(annotator_model_config: BaseModelConfig):
    """Tests if AnnotatorModel correctly unfreezes the base model when required."""
    model = BaseModel(config=annotator_model_config)
    model.set_freeze_base_model(True)
    model.set_freeze_base_model(False)
    assert all(param.requires_grad for name, param in model._base_model.named_parameters()), "Some are still frozen!"


def test_annotator_model_forward(annotator_model_config: BaseModelConfig):
    """Tests if AnnotatorModel's forward pass correctly calls the base model."""
    model = BaseModel(config=annotator_model_config)

    # Create dummy input
    dummy_input = {
        "input_ids": torch.randint(0, 100, (1, 128)),  # batch_size=1, sequence_length=128
        "attention_mask": torch.ones(1, 128),
    }

    # Ensure forward method runs without error
    output = model(**dummy_input)
    assert output is not None


def test_multi_target_regression_head():
    """Tests MultiTargetRegressionHead functionality."""
    head = MultiTargetRegressionHead(
        input_dim=768,
        num_prediction_tasks=2,
        num_targets_per_prediction_task=torch.tensor([6, 6]),
    )

    # Dummy input
    dummy_input = torch.randn(4, 768)  # batch_size=4, input_dim=768
    output = head(dummy_input)

    assert output.shape == (4, 2), f"Unexpected shape: {output.shape}"


def test_multi_target_classification_head():
    """Tests MultiTargetClassificationHead functionality."""
    num_prediction_tasks = 2
    max_num_targets_per_prediction_task = 6
    head = MultiTargetClassificationHead(
        input_dim=768,
        num_prediction_tasks=num_prediction_tasks,
        num_targets_per_prediction_task=torch.tensor(
            [max_num_targets_per_prediction_task, max_num_targets_per_prediction_task - 1]
        ),
    )

    dummy_input = torch.randn(4, 768)  # batch_size=4, input_dim=768
    output = head(dummy_input)

    assert output.shape == (
        4,
        max_num_targets_per_prediction_task,
        num_prediction_tasks,
    ), f"Unexpected batch size: {output.shape}"


def test_logit_mask_layer():
    """Tests whether LogitMaskLayer correctly applies masking."""
    # Two tasks with different numbers of targets
    num_targets_per_task = torch.tensor([3, 2], dtype=torch.int64)
    logit_mask_layer = LogitMaskLayer(num_targets_per_task)

    # Create dummy logits
    logits = torch.randn(5, 6)  # batch_size=5, max_num_classes=3*2=6

    masked_logits = logit_mask_layer(logits)
    assert masked_logits.shape == (5, 3, 2), f"Unexpected output shape: {masked_logits.shape}"


def test_regression_scaling_layer():
    """Tests RegressionScalingLayer scaling behavior."""
    scaling_constants = torch.tensor([2, 3], dtype=torch.int64)
    scaling_layer = RegressionScalingLayer(scaling_constants - 1.0)

    # Dummy input
    x = torch.tensor([[0.5, 1.0], [1.5, 2.0]])

    # Forward pass (training mode)
    scaling_layer.train()
    output_train = scaling_layer(x)
    expected_train = torch.tensor([[0.5, 2.0], [1.5, 4.0]])
    assert torch.all(output_train == expected_train), f"Unexpected training output: {output_train}"

    # Forward pass (evaluation mode)
    scaling_layer.eval()
    output_eval = scaling_layer(x)
    expected_eval = torch.tensor([[0.5, 2.0], [1.0, 2.0]])
    assert torch.all(output_eval == expected_eval), f"Unexpected eval output: {output_eval}"


@pytest.fixture
def annotator_model_config(dummy_base_model_path: str, is_regression: bool) -> BaseModelConfig:
    """Fixture for AnnotatorModel configuration with parameterized is_regression."""
    return BaseModelConfig(
        base_model_name_or_path=dummy_base_model_path,
        num_tasks=2,
        num_targets_per_task=[6, 6],
        is_regression=is_regression,
        load_base_model_from_config=False,
    )


@pytest.fixture(params=[True, False])
def is_regression(request):
    return request.param


@pytest.fixture
def dummy_base_model_path() -> Iterable[str]:
    dummy_path = "bert-base-uncased"
    remove = dummy_path.lower() not in MODEL_CLASS_MAP
    MODEL_CLASS_MAP[dummy_path.lower()] = BertForSequenceClassification
    yield dummy_path
    if remove:
        del MODEL_CLASS_MAP[dummy_path.lower()]
