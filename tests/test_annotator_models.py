import torch

from ml_filter.models.annotator_model_head import MultiTargetRegressionHead
from ml_filter.models.embedding_model import EmbeddingRegressionConfig, EmbeddingRegressionModel


def test_embedding_model_initialization():
    """Tests if EmbeddingRegressionModel initializes correctly with the regression head."""
    regression_config = EmbeddingRegressionConfig(
        embedding_dim=768,
        num_tasks=2,
        num_targets_per_task=[3, 4],
        hidden_dim=1000,
    )

    regression_model = EmbeddingRegressionModel(config=regression_config)
    assert isinstance(regression_model.head, MultiTargetRegressionHead)


def test_embedding_model_forward():
    """Tests if EmbeddingRegressionModel forward pass works."""
    config = EmbeddingRegressionConfig(embedding_dim=768, num_tasks=2, num_targets_per_task=[3, 4])

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
    )

    # Dummy input
    dummy_input = torch.randn(4, 768)  # batch_size=4, input_dim=768
    output = head(dummy_input)

    assert output.shape == (4, 2), f"Unexpected shape: {output.shape}"
