from __future__ import annotations

import numpy as np
import torch

from src.backbone.llm import FrozenLLM
from src.modules.action_head import ContinuousActionHead
from src.modules.obs_projector import ObsProjector
from src.utils.text_prompt import centralized_obs_to_text

from .base import BaseAgent


class CentralizedAgent(BaseAgent):
    """Centralized single-model upper bound baseline.

    Sees both speaker and listener observations concatenated into one prompt.
    Produces the listener's movement action directly.
    """

    def __init__(
        self,
        backbone: FrozenLLM,
        obs_projector: ObsProjector,
        action_head: ContinuousActionHead,
    ):
        super().__init__(backbone, obs_projector, centralized_obs_to_text)
        self.action_head = action_head

    def act(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
        **kwargs,
    ) -> dict:
        """obs should be the concatenated speaker+listener observation."""
        z = self.encode_obs(obs)
        z_batch = z.unsqueeze(0)
        out = self.action_head.get_action(z_batch, deterministic=deterministic)
        return {
            "env_action": out["action"].squeeze(0).detach().cpu().numpy(),
            "action": out["action"].squeeze(0),
            "raw_action": out["mean"].squeeze(0),
            "log_prob": out["log_prob"].squeeze(0),
            "value": out["value"].squeeze(0),
            "entropy": out["entropy"].squeeze(0),
        }

    def act_batch(
        self,
        obs_list: list[np.ndarray],
    ) -> dict[str, torch.Tensor]:
        z_batch = self.encode_obs_batch(obs_list)
        return self.action_head.get_action(z_batch)

    def evaluate_batch(
        self,
        obs_list: list[np.ndarray],
        raw_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        z_batch = self.encode_obs_batch(obs_list)
        return self.action_head.evaluate_action(z_batch, raw_actions)
