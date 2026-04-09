from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CommChannel


class DiscreteChannel(CommChannel):
    """Discrete symbol communication with Gumbel-Softmax.

    During training: soft one-hot via Gumbel-Softmax (differentiable).
    During eval: hard one-hot via argmax.
    """

    def __init__(self, input_dim: int, num_symbols: int, tau: float = 1.0):
        super().__init__()
        self.num_symbols = num_symbols
        self.tau = tau
        self.logit_net = nn.Sequential(
            nn.Linear(input_dim, num_symbols * 2),
            nn.GELU(),
            nn.Linear(num_symbols * 2, num_symbols),
        )

    def forward(self, z_sender: torch.Tensor) -> torch.Tensor:
        logits = self.logit_net(z_sender)
        if self.training:
            return F.gumbel_softmax(logits, tau=self.tau, hard=False)
        else:
            idx = logits.argmax(dim=-1)
            return F.one_hot(idx, self.num_symbols).float()

    def message_dim(self) -> int:
        return self.num_symbols
