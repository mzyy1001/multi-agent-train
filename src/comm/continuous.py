from __future__ import annotations

import torch
import torch.nn as nn

from .base import CommChannel


class ContinuousChannel(CommChannel):
    """Unstructured continuous vector channel.

    Same dimension as SSR but uses only a linear projection —
    no bottleneck MLP or LayerNorm. This is the ablation control
    for testing whether SSR structure helps beyond merely using
    a continuous channel.
    """

    def __init__(self, input_dim: int, dim: int):
        super().__init__()
        self.dim = dim
        self.proj = nn.Linear(input_dim, dim)

    def forward(self, z_sender: torch.Tensor) -> torch.Tensor:
        return self.proj(z_sender)

    def message_dim(self) -> int:
        return self.dim
