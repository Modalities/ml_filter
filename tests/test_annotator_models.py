import torch

from ml_filter.models.annotator_model_head import MultiTargetClassificationHead, MultiTargetRegressionHead
from ml_filter.models.annotator_models import AnnotatorModel, LogitMaskLayer, RegressionScalingLayer


def test_annotator_model_initialization(dummy_base_model, regression_head):
    """Tests if AnnotatorModel initializes correctly with a base model and head."""
    model = AnnotatorModel(
        base_model=dummy_base_model,
        freeze_base_model_parameters=False,
        head=regression_head,
    )

    assert model.base_model is not None
    assert isinstance(model.base_model.classifier, MultiTargetRegressionHead)


def test_annotator_model_freezing(dummy_base_model, regression_head):
    """Tests if AnnotatorModel correctly freezes the base model when required."""

    # Create the model with freezing enabled
    model = AnnotatorModel(
        base_model=dummy_base_model,
        freeze_base_model_parameters=True,
        head=regression_head,
    )

    # Ensure that not all base model parameters are frozen
    assert not all(
        param.requires_grad for param in model.base_model.parameters()
    ), "All base model parameters are frozen!"

    # Ensure classifier parameters are trainable
    assert all(
        param.requires_grad for param in model.base_model.classifier.parameters()
    ), "Classifier parameters should be trainable!"

    # Special case for BERT: Ensure pooler parameters are trainable
    if hasattr(model.base_model, "bert") and hasattr(model.base_model.bert, "pooler"):
        assert all(
            param.requires_grad for param in model.base_model.bert.pooler.parameters()
        ), "BERT pooler parameters should be trainable!"

    # Ensure all other parameters are frozen
    assert all(
        not param.requires_grad
        for name, param in model.base_model.named_parameters()
        if "classifier" not in name and "pooler" not in name  # Exclude classifier & pooler from check
    ), "Some non-classifier and non-pooler parameters are still trainable!"


def test_annotator_model_forward(dummy_base_model, regression_head):
    """Tests if AnnotatorModel's forward pass correctly calls the base model."""
    model = AnnotatorModel(
        base_model=dummy_base_model,
        freeze_base_model_parameters=False,
        head=regression_head,
    )

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
    num_targets_per_task = torch.tensor([3, 2])
    logit_mask_layer = LogitMaskLayer(num_targets_per_task)

    # Create dummy logits
    logits = torch.randn(5, 6)  # batch_size=5, max_num_classes=3*2=6

    masked_logits = logit_mask_layer(logits)
    assert masked_logits.shape == (5, 3, 2), f"Unexpected output shape: {masked_logits.shape}"


def test_regression_scaling_layer():
    """Tests RegressionScalingLayer scaling behavior."""
    scaling_constants = torch.tensor([2.0, 3.0])
    scaling_layer = RegressionScalingLayer(scaling_constants)

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
