"""Sparse MLP head for MinkUNet pretraining (not used in OpenPCDet pipeline)."""

from __future__ import annotations

import torch.nn as nn
import MinkowskiEngine as ME


class SMLP(nn.Module):
    """Sparse MLP projection head for self-supervised pretraining.

    Takes the 96-dim output of MinkUNet and projects through a sequence of
    MinkowskiLinear layers. Only used during TARL pretraining, not during
    downstream detection.

    Args:
        dims: List of hidden dimensions. First layer input is hardcoded to 96
            (MinkUNet output channels).
        use_bn: Apply batch normalization after each linear layer.
        use_relu: Apply ReLU activation between (not after last) linear layers.
        use_dropout: Apply dropout after each linear layer.
        use_bias: Use bias in linear layers.
    """

    def __init__(
        self,
        dims: list[int],
        use_bn: bool = False,
        use_relu: bool = False,
        use_dropout: bool = False,
        use_bias: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        last_dim = dims[0]
        counter = 1
        for dim in dims[1:]:
            layers.append(
                ME.MinkowskiLinear(last_dim if counter > 1 else 96, dim, bias=use_bias)
            )
            counter += 1
            if use_bn:
                layers.append(ME.MinkowskiBatchNorm(dim, eps=1e-5, momentum=0.1))
            if (counter < len(dims)) and use_relu:
                layers.append(ME.MinkowskiReLU(inplace=True))
                last_dim = dim
            if use_dropout:
                layers.append(ME.MinkowskiDropout())
        self.clf = nn.Sequential(*layers)

    def forward(self, batch: ME.SparseTensor) -> ME.SparseTensor:
        """Forward pass through sparse MLP layers."""
        return self.clf(batch)
