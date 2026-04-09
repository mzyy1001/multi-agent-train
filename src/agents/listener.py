from __future__ import annotations

import numpy as np
import torch

from src.backbone.llm import FrozenLLM
from src.modules.action_head import ContinuousActionHead
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter

from .base import BaseAgent


class ListenerAgent(BaseAgent):
    """Listener agent: receives message, produces movement action.

    Uses continuous actions with tanh-squashed Gaussian for full
    differentiability through the message.
    """

    def __init__(
        self,
        backbone: FrozenLLM,
        obs_projector: ObsProjector,
        receiver_adapter: ReceiverAdapter,
        action_head: ContinuousActionHead,
        obs_to_text_fn,
    ):
        super().__init__(backbone, obs_projector, obs_to_text_fn)
        self.adapter = receiver_adapter
        self.action_head = action_head

    def act(
        self,
        obs: np.ndarray,
        message: torch.Tensor,
        deterministic: bool = False,
    ) -> dict:
        """Produce movement action conditioned on message."""
        z = self.encode_obs(obs)  # (proj_dim,)
        z_batch = z.unsqueeze(0)  # (1, proj_dim)
        msg_batch = message.unsqueeze(0)  # (1, comm_dim)
        h_tilde = self.adapter(z_batch, msg_batch)  # (1, adapter_hidden)
        out = self.action_head.get_action(h_tilde, deterministic=deterministic)
        return {
            "env_action": out["action"].squeeze(0).detach().cpu().numpy(),
            "action": out["action"].squeeze(0),
            "raw_action": out["mean"].squeeze(0),  # store pre-sigmoid for re-eval
            "log_prob": out["log_prob"].squeeze(0),
            "value": out["value"].squeeze(0),
            "entropy": out["entropy"].squeeze(0),
            "z": z,
        }

    def act_batch(
        self,
        obs_list: list[np.ndarray],
        messages: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Batched forward for PPO re-evaluation."""
        z_batch = self.encode_obs_batch(obs_list)  # (batch, proj_dim)
        h_tilde = self.adapter(z_batch, messages)  # (batch, adapter_hidden)
        out = self.action_head.get_action(h_tilde)
        return out

    def evaluate_batch(
        self,
        obs_list: list[np.ndarray],
        messages: torch.Tensor,
        raw_actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Re-evaluate stored actions with fresh gradients (for PPO update)."""
        z_batch = self.encode_obs_batch(obs_list)  # (batch, proj_dim)
        h_tilde = self.adapter(z_batch, messages)  # (batch, adapter_hidden)
        return self.action_head.evaluate_action(h_tilde, raw_actions)
