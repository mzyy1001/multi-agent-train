from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.nn as nn

from src.backbone.lora_llm import LoRALLM, LoRAHiddenStateCache
from src.modules.obs_projector import ObsProjector


class LoRABaseAgent(nn.Module, ABC):
    """Agent with LoRA-enabled LLM backbone.

    Unlike BaseAgent (which keeps the frozen backbone outside nn.Module),
    this registers the LoRA model as a submodule so its parameters are
    included in the optimizer and receive gradients.

    The gradient path for the speaker:
      PPO loss → ActionHead → Adapter → message → CommChannel → ObsProjector → LoRA LLM
    """

    def __init__(
        self,
        backbone: LoRALLM,
        obs_projector: ObsProjector,
        obs_to_text_fn,
    ):
        super().__init__()
        self.obs_projector = obs_projector
        # Register LoRA backbone as submodule (its params will be in parameters())
        self.backbone = backbone
        self._cache = LoRAHiddenStateCache(backbone, obs_to_text_fn)

    def encode_obs(self, obs: np.ndarray, training: bool = False) -> torch.Tensor:
        """Encode observation through LoRA LLM + trainable projector."""
        h = self._cache.get(obs, training=training).float()
        z = self.obs_projector(h.unsqueeze(0))
        return z.squeeze(0)

    def encode_obs_batch(self, obs_list: list[np.ndarray], training: bool = False) -> torch.Tensor:
        """Batch encode observations."""
        h_batch = self._cache.get_batch(obs_list, training=training).float()
        return self.obs_projector(h_batch)

    def clear_cache(self):
        self._cache.clear()

    @abstractmethod
    def act(self, obs: np.ndarray, **kwargs) -> dict:
        ...
