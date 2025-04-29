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
