import torch
import torch.nn as nn
from transformers.modeling_utils import ModelOutput, PreTrainedModel


class AnnotatorModel(nn.Module):
    """Base Annotator Model that wraps a pre-trained Transformer with a custom head."""

    def __init__(
        self,
        base_model: PreTrainedModel,
        freeze_base_model_parameters: bool,
        head: nn.Module,
    ):
        """Initializes the AnnotatorModel.

        Args:
            base_model (PreTrainedModel): A pre-trained Transformer model.
            free
            head (nn.Module): The custom classification or regression head.
        """
        super().__init__()
        self.base_model = base_model
        self.freeze_base_model_parameters = freeze_base_model_parameters
        self._overwrite_head(head=head)
        if self.freeze_base_model_parameters:
            self._freeze_base_model()

    def _overwrite_head(self, head: nn.Module):
        """Replaces the classifier in the base model with the custom head."""
        if hasattr(self.base_model, "classifier"):
            if hasattr(self.base_model.classifier, "out_proj"):
                self.base_model.classifier.out_proj = head
            else:
                self.base_model.classifier = head
        else:
            raise AttributeError("The base model does not have a 'classifier' attribute.")

    def _freeze_base_model(self):
        """Freezes all base model parameters, so that only the classification head is trainable.

        This function works with any Transformer model that has a `classifier` or
        `out_proj` layer, ensuring that only the classifier is fine-tuned.
        """
        # Freeze all base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False

        # Unfreeze classifier head
        if hasattr(self.base_model, "classifier"):
            for param in self.base_model.classifier.parameters():
                param.requires_grad = True

        # Special case (BERT): Handle cases where the model has an additional `pooler` layer.
        # TODO: check for more general solution
        if hasattr(self.base_model, "bert") and hasattr(self.base_model.bert, "pooler"):
            for param in self.base_model.bert.pooler.parameters():
                param.requires_grad = True

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> ModelOutput:
        """Forward pass through the base model.

        Args:
            input_ids (Tensor | None): Tokenized input IDs of shape (batch_size, sequence_length).
            attention_mask (Tensor | None): Attention mask tensor of shape (batch_size, sequence_length).
            token_type_ids (Tensor | None): Segment embeddings tensor (if applicable to model type).
            labels (Tensor | None): Labels for training (if applicable, e.g., for classification).

        Returns:
            ModelOutput: Output of the pre-trained transformer model.
        """
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=return_dict,
        )


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
