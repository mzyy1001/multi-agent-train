from __future__ import annotations

import torch
import torch.nn as nn

from .base import CommChannel


class SSRChannel(CommChannel):
    """Structured Semantic Representation channel.

    Encodes sender's projected hidden state into a low-dimensional
    structured vector via a bottleneck MLP with LayerNorm.
    The low dimensionality (4/8/16) forces structured compression.
    """

    def __init__(self, input_dim: int, dim: int, normalize: bool = True, **kwargs):
        super().__init__()
        self.dim = dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm = nn.LayerNorm(dim) if normalize else nn.Identity()

    def forward(self, z_sender: torch.Tensor) -> torch.Tensor:
        return self.norm(self.encoder(z_sender))

    def message_dim(self) -> int:
        return self.dim
