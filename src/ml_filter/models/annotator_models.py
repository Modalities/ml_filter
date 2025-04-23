from abc import ABC
from copy import deepcopy

import torch
from transformers import AutoConfig, PretrainedConfig
from transformers.modeling_utils import ModelOutput, PreTrainedModel

from constants import MODEL_CLASS_MAP
from ml_filter.models.annotator_model_head import (
    AnnotatorHead,
    MultiTargetClassificationHead,
    MultiTargetRegressionHead,
)


class AnnotatorConfig(PretrainedConfig):
    """Configuration class for the AnnotatorModel."""

    def __init__(
        self,
        is_regression: bool = True,
        num_tasks: int = 1,
        num_targets_per_task: list[int] = [2],
        base_model_name_or_path: str = "",
        load_base_model_from_config: bool = False,
        **kwargs,
    ):
        """Initializes the AnnotatorConfig.

        Args:
            is_regression (bool): Whether the model is for regression or classification.
            num_tasks (int): Number of prediction tasks.
            num_targets_per_task list[int]: Number of targets per prediction task.
            base_model_name_or_path (str, optional): Path or name of the pre-trained base model.
            load_base_model_from_config (bool, optional): Whether to load the base model from a
                                                          pretrained config or checkpoint.
                                                          Defaults to False.
            **kwargs: Additional keyword arguments for the configuration.
        """
        super().__init__(**kwargs)
        self.is_regression = is_regression
        self.num_tasks = num_tasks
        self.num_targets_per_task = list(num_targets_per_task)
        self.base_model_name_or_path = base_model_name_or_path
        self.load_base_model_from_config = load_base_model_from_config


class AnnotatorModel(PreTrainedModel):
    """Annotator Model that wraps a pre-trained Transformer with a custom head.
    This model is designed to be used for multi-task learning, where each task
    can have a different number of classes.
    The model can be used for both regression and classification tasks.
    """

    config_class = AnnotatorConfig

    def __init__(self, config: AnnotatorConfig):
        """Initializes the AnnotatorModel by loading the base model either from
        an existing checkpoint or from a configuration. Then replace the
        classifier head with a custom one.

        Args:
            config (AnnotatorConfig): The configuration for the model.
        """
        config = deepcopy(config)
        super().__init__(config)
        self._load_base_model(config)
        self._overwrite_head(config)

    def set_freeze_base_model(self, freeze: bool):
        """If enabled, freezes all base model parameters, so that only the classification head is trainable.

        This function works with any Transformer model that has a `classifier` or
        `out_proj` layer, ensuring that only the classifier is fine-tuned.
        """
        # Freeze all base model parameters
        for param in self._base_model.parameters():
            param.requires_grad = not freeze

        # Unfreeze classifier head
        if hasattr(self._base_model, "classifier"):
            for param in self._base_model.classifier.parameters():
                param.requires_grad = True

        # Special case (BERT): Handle cases where the model has an additional `pooler` layer.
        # TODO: check for more general solution
        if hasattr(self._base_model, "bert") and hasattr(self._base_model.bert, "pooler"):
            for param in self._base_model.bert.pooler.parameters():
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
            return_dict (bool | None): Whether to return a ModelOutput object.

        Returns:
            ModelOutput: Output of the pre-trained transformer model.
        """
        return self._base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=return_dict,
        )

    def _load_base_model(self, config: AnnotatorConfig):
        try:
            model_class = MODEL_CLASS_MAP.get(config.base_model_name_or_path.lower())
        except KeyError:
            raise ValueError(
                f"Model class not found for {config.base_model_name_or_path.name}."
                f" Available models: {MODEL_CLASS_MAP.keys()}"
            )
        self._base_model_config = AutoConfig.from_pretrained(config.base_model_name_or_path)
        if config.load_base_model_from_config:
            self._base_model = model_class(self._base_model_config)
        else:
            self._base_model = model_class.from_pretrained(config.base_model_name_or_path)

    def _overwrite_head(self, config: AnnotatorConfig):
        """Replaces the classifier in the base model with the custom head."""
        head = self._build_new_head(config)
        if hasattr(self._base_model, "classifier"):
            if hasattr(self._base_model.classifier, "out_proj"):
                self._base_model.classifier.out_proj = head
            else:
                self._base_model.classifier = head
        else:
            raise AttributeError("The base model does not have a 'classifier' attribute.")

    def _build_new_head(self, config: AnnotatorConfig) -> AnnotatorHead:
        head_cls = MultiTargetRegressionHead if config.is_regression else MultiTargetClassificationHead
        return head_cls(
            input_dim=self._base_model_config.hidden_size,
            num_prediction_tasks=config.num_tasks,
            num_targets_per_prediction_task=torch.tensor(config.num_targets_per_task),
        )
