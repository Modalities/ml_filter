import torch
from transformers import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from ml_filter.models.annotator_model_head import (
    AnnotatorHead,
    MultiTargetClassificationHead,
    MultiTargetRegressionHead,
)


class EmbeddingRegressionConfig(PretrainedConfig):
    """Configuration for embedding-based regression model."""

    def __init__(
        self,
        embedding_dim: int = 768,
        num_tasks: int = 1,
        num_targets_per_task: list[int] = None,
        hidden_dim: int = 1000,
        is_regression: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_tasks = num_tasks
        self.num_targets_per_task = list(num_targets_per_task) if num_targets_per_task else [1]
        self.hidden_dim = hidden_dim
        self.is_regression = is_regression


class EmbeddingRegressionModel(PreTrainedModel):
    """Model for regression or classification tasks using pre-computed embeddings."""

    config_class = EmbeddingRegressionConfig

    def __init__(self, config: EmbeddingRegressionConfig):
        super().__init__(config)
        self.config = config

        # Initialize the classification head
        self.head = self._build_head(config)

    def _build_head(self, config: EmbeddingRegressionConfig) -> AnnotatorHead:
        """Builds the head for regression or classification tasks based on the configuration."""
        head_cls = MultiTargetRegressionHead if config.is_regression else MultiTargetClassificationHead
        head_params = {
            "input_dim": config.embedding_dim,
            "num_prediction_tasks": config.num_tasks,
            "num_targets_per_prediction_task": torch.tensor(config.num_targets_per_task, dtype=torch.int64),
        }
        if config.is_regression:
            head_params["hidden_dim"] = config.hidden_dim

        return head_cls(**head_params)

    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor | None = None,
        return_dict: bool | None = None,
    ) -> SequenceClassifierOutput:
        """Forward pass with pre-computed embeddings."""
        logits = self.head(embeddings)

        return SequenceClassifierOutput(
            logits=logits,
            hidden_states=None,
            attentions=None,
        )
