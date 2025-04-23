import logging
from pathlib import Path

import torch
import torch.nn as nn

from ml_filter.models.annotator_models import AnnotatorModel


class ModelWrapper(nn.Module):
    def __init__(self, model: nn.Module, use_regression: bool, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._model = model
        self._use_regression = use_regression

    def forward(self, *args, **kwargs):
        outputs = self._model(*args, **kwargs)
        if self._use_regression:
            predictions = torch.round(outputs.logits)
        else:
            predictions = torch.argmax(outputs.logits.squeeze(-1), dim=-1)
        return predictions


class ModelFactory:
    @staticmethod
    def load_huggingface_model_checkpoint(
        model_checkpoint_path: Path,
        device: torch.device,
        logger: logging.Logger,
    ) -> nn.Module:
        model = AnnotatorModel.from_pretrained(model_checkpoint_path)
        wrapped_model = ModelWrapper(model, model.config.is_regression)
        wrapped_model.to(device).eval()
        logger.info("Compiling model...")
        compiled_model = torch.compile(wrapped_model, mode="reduce-overhead")
        logger.info("Compiling done.")
        return compiled_model
