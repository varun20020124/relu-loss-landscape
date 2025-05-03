import torch
import torch.nn as nn

class ShallowReLU(nn.Module):
    """
    A simple feedforward neural network with one hidden layer and ReLU activation.
    """
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: applies ReLU to hidden layer, then linear output.
        """
        x = self.hidden_layer(x)
        x = torch.relu(x)
        x = self.output_layer(x)
        return x
