import logging

from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments


class SpearmanEarlyStoppingCallback(TrainerCallback):
    """
    A callback for early stopping based on the Spearman correlation metric during model training.

    Args:
        patience (int): The number of consecutive epochs without significant improvement
                        before stopping training. Default is 5.
        min_delta (float): The minimum change in the monitored metric to qualify as an improvement.
                        Default is 1e-3.
        metric_key (str): The key of the evaluation metric to monitor. Default is "eval_spearman".
                        If set to "eval_val_loss", it will monitor the validation loss instead.

    Attributes:
        patience (int): The patience value provided during initialization.
        min_delta (float): The minimum delta value provided during initialization.
        metric_key (str): The key of the evaluation metric being monitored.
        best_score (float or None): The best score observed for the monitored metric.
                                    Initialized to None.
        bad_epochs (int): The count of consecutive epochs without significant improvement.
        logger (logging.Logger): Logger instance for logging callback events.

    Methods:
        on_evaluate(args, state, control, **kwargs):
            Called during evaluation. Monitors the specified metric and determines whether
            training should stop based on the early stopping criteria.
    """

    def __init__(self, patience=5, min_delta=1e-3, metric_key="eval_spearman"):
        self.patience = patience
        self.min_delta = min_delta
        if metric_key.startswith("eval_"):
            self.metric_key = metric_key
        else:
            self.metric_key = f"eval_{metric_key}"
        self.best_score = None
        self.bad_epochs = 0
        self.logger = logging.getLogger(__name__)

    def on_evaluate(
        self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs
    ) -> TrainerControl:
        metrics = kwargs.get("metrics")
        if metrics is None or self.metric_key not in metrics:
            self.logger.warning(f"Metric '{self.metric_key}' not found in evaluation metrics.")
            return control

        current_score = metrics[self.metric_key]
        self.logger.info(f"Evaluation {self.metric_key}: {current_score:.6f}")

        if self.best_score is None:
            self.best_score = current_score
            return control

        improvement = current_score - self.best_score
        if improvement < self.min_delta:
            self.bad_epochs += 1
            self.logger.info(
                f"No significant improvement (Δ={improvement:.6f} < {self.min_delta}). Bad epochs: {self.bad_epochs}"
            )
        else:
            self.best_score = current_score
            self.bad_epochs = 0
            self.logger.info(f"Improved {self.metric_key} to {current_score:.6f}")

        if self.bad_epochs >= self.patience:
            self.logger.info(f"Early stopping triggered after {self.bad_epochs} bad epochs.")
            control.should_training_stop = True

        return control
