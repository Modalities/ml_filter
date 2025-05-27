from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
import logging

class SpearmanEarlyStoppingCallback(TrainerCallback):
    def __init__(self, patience=5, min_delta=1e-3, metric_key="eval_spearman"):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_key = f"eval_edu/{metric_key}"
        self.best_score = None
        self.bad_epochs = 0
        self.logger = logging.getLogger(__name__)

    def on_evaluate(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
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
            self.logger.info(f"No significant improvement (Δ={improvement:.6f} < {self.min_delta}). Bad epochs: {self.bad_epochs}")
        else:
            self.best_score = current_score
            self.bad_epochs = 0
            self.logger.info(f"Improved {self.metric_key} to {current_score:.6f}")

        if self.bad_epochs >= self.patience:
            self.logger.info(f"Early stopping triggered after {self.bad_epochs} bad epochs.")
            control.should_training_stop = True

        return control
