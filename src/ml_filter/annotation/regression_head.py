from torch import nn
import lightning as L

class RegressionHead(L.LightningModule):
    """
    A simple regression head implemented as a LightningModule.

    This module defines a two-layer neural network with a ReLU activation
    function in between, designed for regression tasks. 

    Args:
        input_dim (int): The dimensionality of the input features.
        hidden_dim (int): The number of neurons in the hidden layer.
        learning_rate (float): The learning rate for the optimizer.
        train_sets: This parameter appears to be unused in the provided code
                    snippet for `RegressionHead`. It's possible it's intended
                    for use in a larger training setup or an optimizer
                    configuration not shown.
    """

    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        # First linear layer transforms input_dim features to hidden_dim features.
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        # ReLU activation function to introduce non-linearity.
        self.relu = nn.ReLU()
        # Second linear layer transforms hidden_dim features to a single output (for regression).
        self.layer2 = nn.Linear(hidden_dim, 1)
        

    def forward(self, x):
        """
        Defines the forward pass of the regression head.

        Args:
            x (torch.Tensor): The input tensor to the model.

        Returns:
            torch.Tensor: The output of the regression head.
        """
        # Pass the input through the first linear layer.
        x = self.layer1(x)
        # Apply the ReLU activation function.
        x = self.relu(x)
        # Pass the result through the second linear layer to get the final output.
        x = self.layer2(x)
        return x