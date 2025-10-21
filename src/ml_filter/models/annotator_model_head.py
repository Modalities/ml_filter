from abc import ABC, abstractmethod

import torch
import torch.nn as nn


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
    """Head for multi-target regression tasks."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_prediction_tasks: int,
        use_bias: bool = True,
    ):
        """Initializes the multi-target regression head.

        Args:
            input_dim (int): Number of input features from the encoder.
            num_prediction_tasks (int): Number of regression tasks.
            use_bias (bool, optional): Whether to include a bias term in the linear layer. Defaults to True.
        """
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=use_bias),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_prediction_tasks, bias=use_bias),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies the regression head to the input tensor.

        Args:
            x (Tensor): Input tensor of shape `(batch_size, input_dim)`.

        Returns:
            Tensor: Regression output tensor of shape `(batch_size, num_prediction_tasks)`.
        """
        return self.mlp(x)
