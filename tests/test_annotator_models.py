import pytest
import torch

from ml_filter.models.annotator_model_head import (
    LogitMaskLayer,
    MultiTargetClassificationHead,
    MultiTargetRegressionHead,
    RegressionScalingLayer,
)
from ml_filter.models.embedding_model import EmbeddingRegressionConfig, EmbeddingRegressionModel


def test_embedding_model_initialization():
    """Tests if EmbeddingRegressionModel initializes correctly with custom heads."""
    # Test regression model
    regression_config = EmbeddingRegressionConfig(
        embedding_dim=768, num_tasks=2, num_targets_per_task=[3, 4], hidden_dim=1000, is_regression=True
    )

    regression_model = EmbeddingRegressionModel(config=regression_config)
    assert isinstance(regression_model.head, MultiTargetRegressionHead)

    # Test classification model
    classification_config = EmbeddingRegressionConfig(
        embedding_dim=768, num_tasks=2, num_targets_per_task=[3, 4], hidden_dim=1000, is_regression=False
    )

    classification_model = EmbeddingRegressionModel(config=classification_config)
    assert isinstance(classification_model.head, MultiTargetClassificationHead)


def test_embedding_model_forward():
    """Tests if EmbeddingRegressionModel forward pass works."""
    config = EmbeddingRegressionConfig(embedding_dim=768, num_tasks=2, num_targets_per_task=[3, 4], is_regression=True)

    model = EmbeddingRegressionModel(config=config)

    # Create dummy embeddings and labels
    embeddings = torch.randn(4, 768)  # batch_size=4, embedding_dim=768
    labels = torch.randn(4, 2)  # batch_size=4, num_tasks=2

    # Forward pass
    output = model(embeddings=embeddings, labels=labels)

    assert output.logits is not None
    assert output.logits.shape == (4, 2)  # batch_size=4, num_tasks=2


def test_multi_target_regression_head():
    """Tests MultiTargetRegressionHead functionality."""
    head = MultiTargetRegressionHead(
        input_dim=768,
        hidden_dim=1000,
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


@pytest.fixture(params=[True, False])
def is_regression(request):
    return request.param
