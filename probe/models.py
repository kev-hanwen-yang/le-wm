import torch


# Component: probe model definitions.
#
# This module owns the neural network architectures trained on top of frozen
# LeWM embeddings. Probe models take cached embeddings as input and predict one
# physical quantity target. The encoder remains frozen; only these probe model
# parameters are optimized during supervised probe training.
#
# Current model:
# - LinearRegressionProbe: one affine layer, y_hat = W @ embedding + b.


class LinearRegressionProbe(torch.nn.Linear):
    """Linear regression probe for decoding physical quantities from embeddings."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__(input_dim, output_dim)
