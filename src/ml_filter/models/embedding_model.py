import torch
from transformers import PretrainedConfig
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel

from ml_filter.models.annotator_model_head import AnnotatorHead, MultiTargetRegressionHead


class EmbeddingRegressionConfig(PretrainedConfig):
    """
    Configuration class for an embedding-based regression model.

    Attributes:
        embedding_dim (int): The dimensionality of the embedding vector. Default is 768.
        num_tasks (int): The number of tasks for the regression model. Default is 1.
        num_targets_per_task (list[int]): A list specifying the number of target values per task.
            If not provided, defaults to [1].
        hidden_dim (int): The dimensionality of the hidden layer in the model. Default is 1000.
        **kwargs: Additional keyword arguments passed to the parent PretrainedConfig class.
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        num_tasks: int = 1,
        num_targets_per_task: list[int] = None,
        hidden_dim: int = 1000,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.embedding_dim = embedding_dim
        self.num_tasks = num_tasks
        self.num_targets_per_task = list(num_targets_per_task) if num_targets_per_task else [1]
        self.hidden_dim = hidden_dim


class EmbeddingRegressionModel(PreTrainedModel):
    """
    Model for regression tasks using pre-computed embeddings.

    Attributes:
        config_class (EmbeddingRegressionConfig): The configuration class used for the model.
        config (EmbeddingRegressionConfig): The configuration instance containing model parameters.
        head (AnnotatorHead): The task-specific head for regression.
    """

    config_class = EmbeddingRegressionConfig

    def __init__(self, config: EmbeddingRegressionConfig):
        super().__init__(config)
        self.config = config

        # Initialize the regression head
        self.head = self._build_head(config)

    def _build_head(self, config: EmbeddingRegressionConfig) -> AnnotatorHead:
        """Builds the regression head based on the configuration."""
        head_params = {
            "input_dim": config.embedding_dim,
            "num_prediction_tasks": config.num_tasks,
            "num_targets_per_prediction_task": torch.tensor(config.num_targets_per_task, dtype=torch.int64),
            "hidden_dim": config.hidden_dim,
        }

        return MultiTargetRegressionHead(**head_params)

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
