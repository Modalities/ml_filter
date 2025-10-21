import logging
from typing import Final

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


def get_logger(name: str, level: int | None = logging.INFO, format: str | None = None) -> logging.Logger:
    """Create a logger instance with the specified name."""
    logging_args: dict[str, object] = {"level": level}
    if format:
        logging_args["format"] = format
    logging.basicConfig(**logging_args)
    return logging.getLogger(name)


class SuppressTransformersFLOPWarning:
    """Suppress the noisy FLOP warning emitted by Transformers when using embeddings."""

    _LOGGER_NAME: Final[str] = "transformers.modeling_utils"
    _WARNING_PREFIX: Final[str] = "Could not estimate the number of tokens of the input"

    class _Filter(logging.Filter):
        def __init__(self, user_logger: logging.Logger):
            super().__init__()
            self._user_logger = user_logger
            self._has_informed_user = False

        def filter(self, record: logging.LogRecord) -> bool:
            message = record.getMessage()
            if message.startswith(SuppressTransformersFLOPWarning._WARNING_PREFIX):
                if not self._has_informed_user:
                    self._user_logger.debug(
                        "Transformers cannot estimate FLOPs for pre-computed embeddings; "
                        "skipping FLOP reporting (expected behaviour)."
                    )
                    self._has_informed_user = True
                return False
            return True

    @staticmethod
    def install(user_logger: logging.Logger) -> None:
        """Attach the filter to the transformers modeling logger."""
        logging.getLogger(SuppressTransformersFLOPWarning._LOGGER_NAME).addFilter(
            SuppressTransformersFLOPWarning._Filter(user_logger)
        )


class EvaluationSplitLoggerCallback(TrainerCallback):
    """Log the name of the evaluation split whenever the Trainer performs evaluation."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        metrics = kwargs.get("metrics") or {}
        split_name = self._infer_split_name(metrics)
        if split_name:
            self.logger.info("Finished evaluation for split '%s'.", split_name)
        else:
            self.logger.info("Finished evaluation (split could not be inferred).")
        return control

    @staticmethod
    def _infer_split_name(metrics: dict[str, float]) -> str | None:
        for key in metrics.keys():
            if not key.startswith("eval_"):
                continue
            remainder = key[len("eval_") :]
            name_fragment = remainder.split("/", 1)[0]
            return name_fragment.split("_", 1)[0]
        return None


__all__ = [
    "get_logger",
    "SuppressTransformersFLOPWarning",
    "EvaluationSplitLoggerCallback",
]
