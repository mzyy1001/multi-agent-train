from __future__ import annotations

import torch
import torch.nn as nn

from .base import CommChannel


class SSRv2Channel(CommChannel):
    """SSR v2: Improved Structured Semantic Representation channel.

    Key improvements over SSR v1:
    1. Optional LayerNorm (can be disabled for ablation)
    2. Residual connection from input projection to output
    3. Deeper bottleneck with configurable expansion factor
    4. Optional dropout for regularization

    These changes address the finding that SSR v1 underperformed discrete
    communication, potentially due to LayerNorm removing magnitude information
    and the bottleneck being too restrictive.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        normalize: bool = False,
        residual: bool = True,
        expansion: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = dim
        self.residual = residual

        # Input projection (for residual connection)
        self.input_proj = nn.Linear(input_dim, dim)

        # Bottleneck MLP
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim * expansion),
            nn.GELU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(dim * expansion, dim),
        )

        self.norm = nn.LayerNorm(dim) if normalize else nn.Identity()

    def forward(self, z_sender: torch.Tensor) -> torch.Tensor:
        h = self.encoder(z_sender)
        if self.residual:
            h = h + self.input_proj(z_sender)
        return self.norm(h)

    def message_dim(self) -> int:
        return self.dim
