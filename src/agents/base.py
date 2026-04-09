from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from src.backbone.llm import FrozenLLM, HiddenStateCache
from src.modules.obs_projector import ObsProjector


class BaseAgent(nn.Module, ABC):
    """Abstract agent combining a frozen LLM backbone with trainable modules.

    The backbone is NOT registered as a submodule (frozen, not in parameters()).
    Only the projector and downstream heads are trainable.
    """

    def __init__(
        self,
        backbone: FrozenLLM,
        obs_projector: ObsProjector,
        obs_to_text_fn,
    ):
        super().__init__()
        self.obs_projector = obs_projector
        # backbone is NOT an nn.Module attribute — kept as plain attribute
        self._backbone = backbone
        self._cache = HiddenStateCache(backbone, obs_to_text_fn)

    def encode_obs(self, obs: np.ndarray) -> torch.Tensor:
        """Encode observation through frozen LLM + trainable projector.

        Returns z_i = E_i(h_i) with gradient tracking through the projector.
        """
        h = self._cache.get(obs).float()  # (hidden_size,), cast to float32
        z = self.obs_projector(h.unsqueeze(0))  # (1, proj_dim)
        return z.squeeze(0)

    def encode_obs_batch(self, obs_list: list[np.ndarray]) -> torch.Tensor:
        """Batch encode observations. Returns (batch, proj_dim)."""
        h_batch = self._cache.get_batch(obs_list).float()  # (batch, hidden_size)
        return self.obs_projector(h_batch)  # (batch, proj_dim)

    def clear_cache(self):
        self._cache.clear()

    @abstractmethod
    def act(self, obs: np.ndarray, **kwargs) -> dict:
        ...
