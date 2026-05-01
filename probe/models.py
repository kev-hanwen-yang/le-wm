import torch


# Component: probe model definitions.
#
# This module owns the neural network architectures trained on top of frozen
# LeWM embeddings. Probe models take cached embeddings as input and predict one
# physical quantity target. The encoder remains frozen; only these probe model
# parameters are optimized during supervised probe training.
#
# Current models:
# - LinearRegressionProbe: one affine layer, y_hat = W @ embedding + b.
# - MLPRegressionProbe: a small nonlinear decoder for testing whether the
#   physical quantity is present but not linearly accessible.


class LinearRegressionProbe(torch.nn.Linear):
    """Linear regression probe for decoding physical quantities from embeddings."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)


class MLPRegressionProbe(torch.nn.Module):
    """Small MLP regression probe for nonlinear physical-quantity decoding."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1")

        layers = []
        current_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(torch.nn.Linear(current_dim, hidden_dim))
            layers.append(torch.nn.ReLU())
            if dropout > 0:
                layers.append(torch.nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(torch.nn.Linear(current_dim, output_dim))
        self.net = torch.nn.Sequential(*layers)

    def forward(self, embeddings):
        return self.net(embeddings)
