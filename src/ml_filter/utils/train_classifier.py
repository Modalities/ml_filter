import torch
from torch import Tensor
from transformers import Trainer, TrainingArguments
from transformers import XLMRobertaForSequenceClassification

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
    def __init__(self, config, num_metrics=1, num_classes_per_metric=None, regression=False):
        super().__init__(config)
        
        # Get the embedding size from the dense layer
        embedding_size = self.classifier.dense.in_features
        
        if regression:
            self.classifier.out_proj = torch.nn.Sequential(
                torch.nn.Linear(embedding_size, num_metrics, bias=True),
                RegressionScalingLayer(num_classes_per_metric),
            )
        else:
            self.num_labels = num_metrics * max(num_classes_per_metric)
            self.classifier.out_proj = torch.nn.Sequential(
                torch.nn.Linear(embedding_size, self.num_labels, bias=True),
                LogitMaskLayer(num_classes_per_metric),
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