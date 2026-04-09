from __future__ import annotations

import torch
import torch.nn as nn


class ReceiverAdapter(nn.Module):
    """Fuses receiver's hidden state with incoming message.

    D_j(z_j, m_i) -> h_tilde
    Concatenation + MLP.
    """

    def __init__(
        self,
        hidden_dim: int,
        message_dim: int,
        output_dim: int,
        adapter_hidden: int,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim + message_dim, adapter_hidden),
            nn.LayerNorm(adapter_hidden),
            nn.GELU(),
            nn.Linear(adapter_hidden, output_dim),
        )

    def forward(
        self,
        h_receiver: torch.Tensor,
        message: torch.Tensor,
    ) -> torch.Tensor:
        """
        h_receiver: (batch, hidden_dim)
        message: (batch, message_dim)
        returns: (batch, output_dim)
        """
        return self.net(torch.cat([h_receiver, message], dim=-1))
