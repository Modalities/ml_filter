from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from ml_filter.models.annotator_models import LogitMaskLayer, RegressionScalingLayer


class AnnotatorHead(nn.Module, ABC):
    """Abstract base class for annotator heads."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the head's transformation.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Output tensor.
        """
        raise NotImplementedError("AnnotatorHead subclasses must implement the forward method.")


class MultiTargetRegressionHead(AnnotatorHead):
    """Head for multi-target regression tasks.

    This module consists of:
    - A linear layer to map input embeddings to the regression outputs.
    - A scaling layer (`RegressionScalingLayer`) to handle multiple regression outputs with different scales.

    Attributes:
        linear (nn.Linear): A fully connected layer that produces raw regression outputs.
        scaling (RegressionScalingLayer): A scaling layer to normalize the outputs.
    """

    def __init__(
        self,
        input_dim: int,
        num_prediction_tasks: int,
        num_targets_per_prediction_task: torch.Tensor,
        use_bias: bool = True,
    ):
        """Initializes the multi-target regression head.

        Args:
            input_dim (int): Number of input features from the encoder.
            num_prediction_tasks (int): Number of regression tasks.
            num_targets_per_prediction_task (Tensor): A tensor defining the number of classes per output
                (used for normalization and scaling).
            use_bias (bool, optional): Whether to include a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.linear = nn.Linear(input_dim, num_prediction_tasks, bias=use_bias)
        self.scaling = RegressionScalingLayer(num_targets_per_prediction_task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the regression head to the input tensor.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, input_dim)`.

        Returns:
            Tensor: Scaled regression output tensor of shape `(batch_size, num_targets)`.
        """
        return self.scaling(self.linear(x))


class MultiTargetClassificationHead(AnnotatorHead):
    """
    Head for multi-target classification tasks.

    This module consists of:
    - A linear layer that projects the input embeddings into a large logit space.
    - A logit mask layer (`LogitMaskLayer`) that ensures each classification task
      has the correct number of output classes.

    Attributes:
        linear (nn.Linear): Fully connected layer that produces raw logits.
        logit_mask (LogitMaskLayer): Layer that applies a mask to the logits based on the number of classes.
    """

    def __init__(
        self,
        input_dim: int,
        num_prediction_tasks: int,
        num_targets_per_prediction_task: torch.Tensor,
        use_bias: bool = True,
    ):
        """Initializes the classification head.

        Args:
            input_dim (int): Number of input features from the encoder.
            num_prediction_tasks (int): Number of classification tasks (each task has its own output).
            num_targets_per_prediction_task (Tensor): A tensor containing the number of classes per prediction task.
            use_bias (bool, optional): Whether to include a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        total_logits = num_prediction_tasks * num_targets_per_prediction_task.max().item()  # Safer way to get max value
        self.linear = nn.Linear(input_dim, total_logits, bias=use_bias)
        self.logit_mask = LogitMaskLayer(num_targets_per_prediction_task)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the classification head to the input tensor.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, input_dim)`.

        Returns:
            Tensor: Processed logits of shape `(batch_size, total_logits)`, after applying the logit mask.
        """
        return self.logit_mask(self.linear(x))
