from __future__ import annotations

import torch
import torch.nn as nn


class ObsProjector(nn.Module):
    """Projects frozen LLM hidden state to a fixed-size latent vector.

    E_i: R^{hidden_size} -> R^{output_dim}
    Two-layer MLP with LayerNorm and GELU.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: (batch, hidden_size) -> z: (batch, output_dim)"""
        return self.net(h)
