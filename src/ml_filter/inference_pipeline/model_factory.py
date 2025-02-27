import logging
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification

from ml_filter.utils.train_classifier import AutoModelForMultiTargetClassification


class ModelFactory:
    @staticmethod
    def load_huggingface_model_checkpoint(
        model_checkpoint_path: Path,
        model_type: str,
        num_regressor_outputs: int,
        num_classes_per_output: list[int],
        use_regression: bool,
        device: torch.device,
        logger: logging.Logger,
    ) -> torch.nn.Module:
        model_args = {
            "model_type": model_type,
            "num_regressor_outputs": num_regressor_outputs,
            "num_classes_per_output": torch.tensor(num_classes_per_output),
            "regression": use_regression,
        }

        try:
            model = AutoModelForMultiTargetClassification.from_pretrained(model_checkpoint_path, **model_args)
        except NotImplementedError:
            logger.warning(
                f"Custom model architecture for {model_checkpoint_path=} not implemented, falling back to AutoModel..."
            )
            model = AutoModelForSequenceClassification.from_pretrained(
                model_checkpoint_path,
                num_labels=num_classes_per_output[0],
            )
        model.to(device).eval()
        return model
