from __future__ import annotations

import torch

from .base import CommChannel


class NoChannel(CommChannel):
    """No communication baseline. Returns a zero vector."""

    def __init__(self, dim: int = 1):
        super().__init__()
        self.dim = dim

    def forward(self, z_sender: torch.Tensor) -> torch.Tensor:
        batch = z_sender.shape[0]
        return torch.zeros(batch, self.dim, device=z_sender.device)

    def message_dim(self) -> int:
        return self.dim
