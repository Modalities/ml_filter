from typing import Iterable

import pytest
import torch
from torch import nn
from transformers import BertForSequenceClassification

from constants import MODEL_CLASS_MAP
from ml_filter.models.annotator_model_head import (
    LogitMaskLayer,
    MultiTargetClassificationHead,
    MultiTargetRegressionHead,
    RegressionScalingLayer,
)
from ml_filter.models.base_model import BaseModel, BaseModelConfig
from ml_filter.models.embedding_model import EmbeddingRegressionConfig, EmbeddingRegressionModel


def test_base_model_initialization(base_model_config: BaseModelConfig):
    """Tests if BaseModel initializes correctly with a base transformer (no custom head)."""
    model = BaseModel(config=base_model_config)

    assert model._base_model is not None
    # BaseModel should keep the original classifier (no custom head replacement)
    assert hasattr(model._base_model, "classifier")
    # The classifier should be the original Linear layer from the transformer
    assert isinstance(model._base_model.classifier, nn.Linear)


def test_base_model_embedding_extraction(base_model_config: BaseModelConfig):
    """Tests if BaseModel can extract embeddings."""
    from omegaconf import DictConfig

    model = BaseModel(config=base_model_config)

    # Create dummy config for embedding extraction
    embedding_config = DictConfig({"embedding": {"normalize_embeddings": True}})

    # Create dummy input
    dummy_input = {
        "input_ids": torch.randint(0, 100, (2, 128)),  # batch_size=2, sequence_length=128
        "attention_mask": torch.ones(2, 128),
    }

    # Extract embeddings
    embeddings = model.extract_embeddings(config=embedding_config, **dummy_input)

    assert embeddings is not None
    assert embeddings.shape[0] == 2  # batch size
    assert embeddings.shape[1] > 0  # embedding dimension


def test_base_model_freezing(base_model_config: BaseModelConfig):
    """Tests if BaseModel correctly freezes the base model when required."""
    # Create the model with freezing enabled
    model = BaseModel(config=base_model_config)
    model.set_freeze_base_model(True, True)

    # Ensure that not all base model parameters are frozen
    assert not all(
        param.requires_grad for param in model._base_model.parameters()
    ), "All base model parameters are frozen!"

    # Ensure classifier parameters are trainable
    assert all(
        param.requires_grad for param in model._base_model.classifier.parameters()
    ), "Classifier parameters should be trainable!"

    # Special case for BERT: Ensure pooler parameters are FROZEN (not trainable)
    if hasattr(model._base_model, "bert") and hasattr(model._base_model.bert, "pooler"):
        assert all(
            not param.requires_grad for param in model._base_model.bert.pooler.parameters()
        ), "BERT pooler parameters should be frozen!"

    # Ensure all other parameters (except classifier) are frozen
    assert all(
        not param.requires_grad
        for name, param in model._base_model.named_parameters()
        if "classifier" not in name  # Only exclude classifier from check (pooler should be frozen)
    ), "Some non-classifier parameters are still trainable!"


def test_base_model_unfreezing(base_model_config: BaseModelConfig):
    """Tests if BaseModel correctly unfreezes the base model when required."""
    model = BaseModel(config=base_model_config)
    model.set_freeze_base_model(True, True)
    model.set_freeze_base_model(False, False)

    # All parameters should be trainable
    assert all(param.requires_grad for param in model._base_model.parameters()), "All parameters should be trainable!"


def test_base_model_forward(base_model_config: BaseModelConfig):
    """Tests if BaseModel's underlying transformer works."""
    model = BaseModel(config=base_model_config)

    dummy_input = {
        "input_ids": torch.randint(1, 1000, (1, 128)),
        "attention_mask": torch.ones(1, 128, dtype=torch.long),
    }

    output = model._base_model(**dummy_input)
    assert output is not None


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


@pytest.fixture
def annotator_model_config(dummy_base_model_path: str, is_regression: bool) -> BaseModelConfig:
    """Fixture for AnnotatorModel configuration with parameterized is_regression."""
    return BaseModelConfig(
        base_model_name_or_path=dummy_base_model_path,
        num_tasks=2,
        num_targets_per_task=[6, 6],
        is_regression=is_regression,
        load_base_model_from_config=False,
        loading_params={
            "trust_remote_code": False,
        },
    )


@pytest.fixture
def base_model_config(dummy_base_model_path: str) -> BaseModelConfig:
    """Fixture for BaseModel configuration (embedding extraction only)."""
    return BaseModelConfig(
        base_model_name_or_path=dummy_base_model_path,
        num_tasks=2,
        num_targets_per_task=[6, 6],
        is_regression=True,  # This doesn't matter for BaseModel
        freeze_base_model_parameters=True,  # Add option to freeze encoder
        freeze_pooling_layer_params=True,  # Applicable on bert like models
        load_base_model_from_config=False,
        loading_params={
            "trust_remote_code": False,
        },
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
