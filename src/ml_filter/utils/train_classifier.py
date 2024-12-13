import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from torch import Tensor
from transformers import (
    BertForSequenceClassification,
    Trainer,
    TrainingArguments,
    XLMRobertaForSequenceClassification,
    XLMRobertaXLForSequenceClassification,
)


class DocumentClassifier:
    def __init__(self, model):
        self.model = model

    def classify_long_document(self, document, max_length=512, stride=256):
        inputs = self.model.model.tokenize([document], truncation=False, return_tensors="pt")
        input_ids = inputs["input_ids"][0]

        chunks = []
        for i in range(0, len(input_ids), stride):
            chunk = input_ids[i : i + max_length]
            chunks.append(chunk)
            if len(chunk) < max_length:
                break

        self.model.eval()
        predictions = []
        with torch.no_grad():
            for chunk in chunks:
                chunk = chunk.unsqueeze(0)
                embeddings = self.model.model.encode(chunk, convert_to_tensor=True)
                logits = self.model.classifier(embeddings)
                predictions.append(logits)

        final_prediction = torch.mean(torch.stack(predictions), dim=0)
        return torch.argmax(final_prediction, dim=1).item()

    def classify_documents(self, docs):
        self.model.eval()
        with torch.no_grad():
            logits = self.model(docs)
            predictions = torch.argmax(logits, dim=1)
        return predictions

    def train(self, train_dataset, output_dir="./results", epochs=3, batch_size=8):
        training_args = TrainingArguments(
            output_dir=output_dir,
            evaluation_strategy="epoch",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=10,
            save_steps=10,
            save_total_limit=2,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            tokenizer=None,  # SentenceTransformer model handles tokenization
        )

        trainer.train()

    def encode_documents(self, docs):
        return self.model.model.encode(docs, convert_to_tensor=True)


class LogitMaskLayer(torch.nn.Module):
    """
    Applies mask to multi-target classifier, for targets with different amounts of classes.
    For example, if we have one target with 3 classes and one target with 2 classes,
    the classifier Linear layer outputs 6 = max(3, 2) * len([3, 2]) logits.
    These are reshaped to (3, 2), and then a mask [[0, 0], [0, 0], [0, -inf]] is added to the logits.
    """

    def __init__(self, num_classes_per_output: Tensor):
        """
        Args:
            num_classes_per_output (Tensor): 1D int/long Tensor, number of classes for each target
        """
        super().__init__()
        self.num_classes_per_output = num_classes_per_output

        self.max_num_classes_per_output = int(max(self.num_classes_per_output))
        self.num_regressor_outputs = self.num_classes_per_output.shape[0]

        self.raw_logit_mask = (
            torch.arange(self.max_num_classes_per_output).repeat(self.num_regressor_outputs, 1).T
            < self.num_classes_per_output
        )
        # use a small value instead of -inf for numerical stability
        self.logit_mask = torch.nn.Parameter((self.raw_logit_mask + 1e-45).log(), requires_grad=False)
        self.mask_shape = self.logit_mask.shape

    def forward(self, x: Tensor) -> Tensor:
        """Reshape logits from linear layer and apply mask.

        Args:
            x (Tensor): shape (batch_size, max_num_classes_per_output * num_regressor_outputs)

        Returns:
            Tensor: shape (batch_size, max_num_classes_per_output, num_regressor_outputs)
        """
        return x.view(-1, *self.mask_shape) + self.logit_mask


class RegressionScalingLayer(torch.nn.Module):
    """
    A PyTorch module that scales regression outputs with clamping during evaluation.

    This layer performs two main functions:
    1. During training: Scales the input tensor without clamping to preserve gradients
    2. During evaluation: Clamps the input tensor to [0, 1] and then scales it

    Attributes:
        scaling_constants (torch.nn.Parameter): A tensor of scaling constants,
            initialized by subtracting 1.0 from the input and set as non-trainable
    """

    def __init__(self, scaling_constants: Tensor):
        """
        Initialize the RegressionScalingLayer.

        Args:
            scaling_constants (Tensor): Tensor used for scaling regression outputs (non-trainable).
                The values are detached, cloned, and adjusted by subtracting 1.0.
                For a target with n_classes classes, valid ground truth values are the integers 0, 1, ..., n_classes-1
        """
        super().__init__()
        self.scaling_constants = torch.nn.Parameter(scaling_constants.detach().clone() - 1.0, requires_grad=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Apply scaling to the input tensor, with different behavior in training and evaluation modes.

        During training, simply multiplies the input by scaling constants.
        During evaluation, clamps the input to [0, 1] before scaling.

        Args:
            x (Tensor): Input tensor of shape (batch_size, num_regressor_outputs)

        Returns:
            Tensor: Scaled output tensor of shape (batch_size, num_regressor_outputs)
        """
        if self.training:  # if training, don't clamp to preserve gradient
            return x * self.scaling_constants
        else:  # clamp to [0, 1] during eval
            return torch.clamp(x, 0.0, 1.0) * self.scaling_constants


class XLMRobertaForMultiTargetClassification(XLMRobertaForSequenceClassification):
    def __init__(self, config, num_regressor_outputs=1, num_classes_per_output=None, regression=False):
        """
        Args:
            config: Model configuration
            num_regressor_outputs: Number of outputs (either regression targets or classification targets)
            num_classes_per_output: Tensor containing the number of classes for each output
            regression: If True, use regression head, otherwise use classification head
        """
        super().__init__(config)

        # Get the embedding size from the dense layer
        embedding_size = self.classifier.dense.in_features

        if regression:
            self.classifier.out_proj = MultiTargetRegressionHead(
                in_features=embedding_size,
                num_outputs=num_regressor_outputs,
                num_classes_per_output=num_classes_per_output,
            )
        else:
            self.classifier.out_proj = MultiTargetClassificationHead(
                in_features=embedding_size,
                num_outputs=num_regressor_outputs,
                num_classes_per_output=num_classes_per_output,
            )


class XLMRobertaXLForMultiTargetClassification(XLMRobertaXLForSequenceClassification):
    def __init__(self, config, num_regressor_outputs=1, num_classes_per_output=None, regression=False):
        """
        Args:
            config: Model configuration
            num_regressor_outputs: Number of outputs (either regression targets or classification targets)
            num_classes_per_output: Tensor containing the number of classes for each output
            regression: If True, use regression head, otherwise use classification head
        """
        super().__init__(config)

        # Get the embedding size from the dense layer
        embedding_size = self.classifier.dense.in_features

        if regression:
            self.classifier.out_proj = MultiTargetRegressionHead(
                in_features=embedding_size,
                num_outputs=num_regressor_outputs,
                num_classes_per_output=num_classes_per_output,
            )
        else:
            self.classifier.out_proj = MultiTargetClassificationHead(
                in_features=embedding_size,
                num_outputs=num_regressor_outputs,
                num_classes_per_output=num_classes_per_output,
            )


class MultiTargetRegressionHead(torch.nn.Module):
    """Head for multi-target regression tasks.

    This module consists of a linear layer followed by a scaling layer to handle
    multiple regression outputs with different scales.
    """

    def __init__(self, in_features: int, num_outputs: int, num_classes_per_output: torch.Tensor):
        """
        Args:
            in_features: Number of input features from the encoder
            num_outputs: Number of regression outputs
            num_classes_per_output: Tensor containing the number of classes for each output
                                  (used for scaling)
        """
        super().__init__()
        self.linear = torch.nn.Linear(in_features, num_outputs, bias=True)
        self.scaling = RegressionScalingLayer(num_classes_per_output)

    def forward(self, x):
        x = self.linear(x)
        x = self.scaling(x)
        return x


class MultiTargetClassificationHead(torch.nn.Module):
    """Head for multi-target classification tasks.

    This module consists of a linear layer followed by a logit mask layer to handle
    multiple classification outputs with different numbers of classes.
    """

    def __init__(self, in_features: int, num_outputs: int, num_classes_per_output: torch.Tensor):
        """
        Args:
            in_features: Number of input features from the encoder
            num_outputs: Number of classification outputs
            num_classes_per_output: Tensor containing the number of classes for each output
        """
        super().__init__()
        total_logits = num_outputs * max(num_classes_per_output)
        self.linear = torch.nn.Linear(in_features, total_logits, bias=True)
        self.logit_mask = LogitMaskLayer(num_classes_per_output)

    def forward(self, x):
        x = self.linear(x)
        x = self.logit_mask(x)
        return x


def compute_metrics_for_single_output(
    labels: np.ndarray, preds: np.ndarray, preds_raw: np.ndarray, thresholds: list
) -> dict:
    """
    Computes evaluation metrics for a specific output.

    Args:
        labels (np.ndarray): Ground truth labels of shape (batch_size,)
        preds (np.ndarray): Predicted class indices of shape (batch_size,)
        preds_raw (np.ndarray): Raw predictions (logits or regression values) of shape (batch_size,) for regression
                               and (batch_size, num_classes) for classification.
        thresholds (list): List of thresholds to use for binary metrics

    Returns:
        dict: Dictionary containing the following metrics:
            - accuracy: Overall classification accuracy
            - f1_weighted: F1 score with weighted averaging
            - f1_micro: F1 score with micro averaging
            - f1_macro: F1 score with macro averaging
            - binary_accuracy_t{t}: Binary accuracy for each threshold t
            - binary_f1_weighted_t{t}: Binary F1 weighted for each threshold t
            - binary_f1_micro_t{t}: Binary F1 micro for each threshold t
            - binary_f1_macro_t{t}: Binary F1 macro for each threshold t
            - mse: Mean squared error between raw predictions and labels
            - mae: Mean absolute error between raw predictions and labels
            - f1_class_{c}: F1 score for each individual class c
    """
    metrics = {}

    # Compute classification metrics
    metrics["classification/accuracy"] = accuracy_score(labels, preds)
    metrics["classification/f1_weighted"] = f1_score(labels, preds, average="weighted")
    metrics["classification/f1_micro"] = f1_score(labels, preds, average="micro")
    metrics["classification/f1_macro"] = f1_score(labels, preds, average="macro")

    # Calculate binary metrics for different thresholds
    for threshold in thresholds:
        # Convert to binary predictions using threshold
        binary_preds = np.where(preds >= threshold, 1, 0)
        binary_labels = np.where(labels >= threshold, 1, 0)

        metrics[f"binary/t{threshold}/accuracy"] = accuracy_score(binary_labels, binary_preds)
        metrics[f"binary/t{threshold}/f1_weighted"] = f1_score(binary_labels, binary_preds, average="weighted")
        metrics[f"binary/t{threshold}/f1_micro"] = f1_score(binary_labels, binary_preds, average="micro")
        metrics[f"binary/t{threshold}/f1_macro"] = f1_score(binary_labels, binary_preds, average="macro")

    # Compute regression-like metrics
    metrics["regression/mse"] = mean_squared_error(labels, preds_raw)
    metrics["regression/mae"] = mean_absolute_error(labels, preds_raw)

    # Add f1 scores for each class
    classes = np.unique(labels)
    classes.sort()
    f1_per_class = f1_score(labels, preds, average=None)
    for i, c in enumerate(classes):
        metrics[f"class_f1/f1_class_{c}"] = f1_per_class[i]

    return metrics


class BertForMultiTargetClassification(BertForSequenceClassification):
    def __init__(self, config, num_regressor_outputs=1, num_classes_per_output=None, regression=False):
        """
        Args:
            config: Model configuration
            num_regressor_outputs: Number of outputs (either regression targets or classification targets)
            num_classes_per_output: Tensor containing the number of classes for each output
            regression: If True, use regression head, otherwise use classification head
        """
        super().__init__(config)

        # Get the embedding size from the classifier
        embedding_size = self.classifier.in_features

        if regression:
            self.classifier = MultiTargetRegressionHead(
                in_features=embedding_size,
                num_outputs=num_regressor_outputs,
                num_classes_per_output=num_classes_per_output,
            )
        else:
            self.classifier = MultiTargetClassificationHead(
                in_features=embedding_size,
                num_outputs=num_regressor_outputs,
                num_classes_per_output=num_classes_per_output,
            )
