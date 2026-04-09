from __future__ import annotations

import numpy as np
import torch

from src.backbone.llm import FrozenLLM
from src.comm.base import CommChannel
from src.modules.action_head import ValueOnlyHead
from src.modules.obs_projector import ObsProjector

from .base import BaseAgent


class SpeakerAgent(BaseAgent):
    """Speaker agent: observes goal, produces message via comm channel.

    The speaker cannot move. Its 'action' is the message sent to the listener.
    It also has a value head for PPO baseline estimation.
    """

    def __init__(
        self,
        backbone: FrozenLLM,
        obs_projector: ObsProjector,
        comm_channel: CommChannel,
        value_head: ValueOnlyHead,
        obs_to_text_fn,
        env_action_dim: int = 3,
    ):
        super().__init__(backbone, obs_projector, obs_to_text_fn)
        self.comm = comm_channel
        self.value_head = value_head
        self.env_action_dim = env_action_dim

    def act(self, obs: np.ndarray, **kwargs) -> dict:
        """Produce message and value estimate from speaker observation."""
        z = self.encode_obs(obs)  # (proj_dim,)
        z_batch = z.unsqueeze(0)  # (1, proj_dim)
        message = self.comm(z_batch).squeeze(0)  # (comm_dim,)
        value = self.value_head(z_batch).squeeze(0)  # scalar
        env_action = np.zeros(self.env_action_dim, dtype=np.float32)
        return {
            "env_action": env_action,
            "message": message,
            "value": value,
            "z": z,
        }

    def act_batch(
        self,
        obs_list: list[np.ndarray],
    ) -> dict[str, torch.Tensor]:
        """Batched forward for PPO re-evaluation."""
        z_batch = self.encode_obs_batch(obs_list)  # (batch, proj_dim)
        messages = self.comm(z_batch)  # (batch, comm_dim)
        values = self.value_head(z_batch)  # (batch,)
        return {
            "message": messages,
            "value": values,
            "z": z_batch,
        }
