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


class RegressionScalingLayer(nn.Module):
    """A PyTorch module that scales regression outputs with clamping during evaluation.

    - **Training Mode**: Multiplies the input tensor by scaling constants (preserving gradients).
    - **Evaluation Mode**: Clamps the input to [0, 1] and then scales it.

    Attributes:
        scaling_constants (torch.nn.Parameter): A tensor of scaling constants,
            initialized by subtracting 1.0 from the input and set as non-trainable.
    """

    def __init__(self, scaling_constants: torch.Tensor):
        """Initializes the scaling layer.

        Args:
            scaling_constants (Tensor): Tensor used for scaling regression outputs (non-trainable).
                The values are adjusted by subtracting 1.0.
                For a target with n_classes, valid values are 0, 1, ..., n_classes-1.
        """
        super().__init__()
        self.register_buffer("scaling_constants", scaling_constants - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scales the input tensor differently during training and evaluation.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_regressor_outputs)

        Returns:
            Tensor: Scaled output tensor of shape (batch_size, num_regressor_outputs)
        """
        if self.training:
            return x * self.scaling_constants
        else:
            return torch.clamp(x, 0.0, 1.0) * self.scaling_constants


class LogitMaskLayer(nn.Module):
    """
    Applies a mask to multi-target classification outputs for tasks with different numbers of classes.

    Example:
        If we have one task with 3 classes and another task with 2 classes:
        - The classifier's output layer produces `max(3, 2) * num_tasks = 6` logits.
        - These logits are reshaped to `(batch_size, 3, 2)`.
        - A mask `[[0, 0], [0, 0], [0, -inf]]` is applied to disable invalid logits.

    Attributes:
        logit_mask (nn.Parameter): Logits mask applied before computing loss.
    """

    def __init__(self, num_targets_per_task: torch.Tensor):
        """
        Args:
            num_classes_per_task (Tensor): A 1D Tensor containing the number of classes for each task.
        """
        super().__init__()

        # Compute max number of classes among all tasks
        self.max_num_targets = num_targets_per_task.max().item()
        self.num_tasks = num_targets_per_task.shape[0]

        # Shape: (max_num_targets, 1)
        target_indices = torch.arange(self.max_num_targets).unsqueeze(1)
        # Shape: (max_targets_per_task, num_tasks)
        self.raw_logit_mask = target_indices < num_targets_per_task.unsqueeze(0)

        # Use a very small value instead of -inf for numerical stability

        # TOOD: Check: Replace: self.register_buffer("logit_mask", (self.raw_logit_mask.float() + 1e-45).log())
        # TODO: Ensure that in mixed precision training, the logit_mask is not converted due to scalar values
        self.register_buffer("logit_mask", torch.where(self.raw_logit_mask, 0.0, float("-inf")))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reshape logits from linear layer and apply mask.

        Args:
            x (Tensor): shape (batch_size, max_num_classes_per_output * num_regressor_outputs)

        Returns:
            Tensor: shape (batch_size, max_num_targets_per_task, num_tasks)
        """
        return x.view(-1, *self.logit_mask.shape) + self.logit_mask
