from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class CommChannel(nn.Module, ABC):
    """Abstract base for all communication channels."""

    @abstractmethod
    def forward(self, z_sender: torch.Tensor) -> torch.Tensor:
        """z_sender: (batch, input_dim) -> message: (batch, message_dim)"""
        ...

    @abstractmethod
    def message_dim(self) -> int:
        """Dimension of the output message vector."""
        ...
