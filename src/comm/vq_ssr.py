from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import CommChannel


class VQSSRChannel(CommChannel):
    """Vector-Quantized SSR: hybrid discrete-continuous communication.

    Combines SSR's differentiable bottleneck with vector quantization (VQ-VIB inspired).
    During training: continuous encoder output + VQ commitment loss for codebook learning.
    During eval: quantized to nearest codebook entry (discrete, interpretable).

    The straight-through estimator (STE) passes gradients through the quantization step,
    preserving the end-to-end differentiability that is SSR's key advantage while
    learning a discrete codebook that aligns with the task's categorical structure.
    """

    def __init__(
        self,
        input_dim: int,
        dim: int,
        num_codes: int = 16,
        commitment_cost: float = 0.25,
        ema_decay: float = 0.99,
    ):
        super().__init__()
        self.dim = dim
        self.num_codes = num_codes
        self.commitment_cost = commitment_cost
        self.ema_decay = ema_decay

        # SSR encoder: same MLP bottleneck as SSRChannel
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

        # Codebook: learnable embedding table
        self.codebook = nn.Embedding(num_codes, dim)
        # Initialize codebook uniformly
        self.codebook.weight.data.uniform_(-1.0 / num_codes, 1.0 / num_codes)

        # EMA tracking for codebook update
        self.register_buffer("_ema_cluster_size", torch.zeros(num_codes))
        self.register_buffer("_ema_embed_sum", torch.zeros(num_codes, dim))
        self._ema_initialized = False

        # Store last VQ loss for external access during training
        self.last_vq_loss = 0.0
        self.last_perplexity = 0.0

    def forward(self, z_sender: torch.Tensor) -> torch.Tensor:
        # Encode through MLP bottleneck
        z_e = self.encoder(z_sender)  # (batch, dim)

        # Compute distances to codebook entries
        # ||z_e - e_k||^2 = ||z_e||^2 + ||e_k||^2 - 2 * z_e . e_k
        distances = (
            z_e.pow(2).sum(dim=-1, keepdim=True)
            + self.codebook.weight.pow(2).sum(dim=-1)
            - 2 * z_e @ self.codebook.weight.t()
        )  # (batch, num_codes)

        # Find nearest codebook entry
        encoding_indices = distances.argmin(dim=-1)  # (batch,)
        z_q = self.codebook(encoding_indices)  # (batch, dim)

        if self.training:
            # VQ losses
            commitment_loss = F.mse_loss(z_e.detach(), z_q) + \
                              self.commitment_cost * F.mse_loss(z_e, z_q.detach())
            self.last_vq_loss = commitment_loss.item()

            # EMA codebook update
            with torch.no_grad():
                encodings = F.one_hot(encoding_indices, self.num_codes).float()
                self._ema_cluster_size = (
                    self.ema_decay * self._ema_cluster_size
                    + (1 - self.ema_decay) * encodings.sum(0)
                )
                embed_sum = encodings.t() @ z_e
                self._ema_embed_sum = (
                    self.ema_decay * self._ema_embed_sum
                    + (1 - self.ema_decay) * embed_sum
                )
                # Laplace smoothing
                n = self._ema_cluster_size.sum()
                cluster_size = (
                    (self._ema_cluster_size + 1e-5)
                    / (n + self.num_codes * 1e-5)
                    * n
                )
                self.codebook.weight.data = self._ema_embed_sum / cluster_size.unsqueeze(1)

                # Perplexity (codebook utilization metric)
                avg_probs = encodings.mean(0)
                self.last_perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10))).item()

            # Straight-through estimator: gradient flows through z_e, output is z_q
            return z_e + (z_q - z_e).detach()
        else:
            # At eval: use quantized codes directly
            return z_q

    def message_dim(self) -> int:
        return self.dim
