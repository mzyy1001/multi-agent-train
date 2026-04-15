from __future__ import annotations

import numpy as np
import torch

from src.backbone.lora_llm import LoRALLM
from src.comm.base import CommChannel
from src.modules.action_head import ValueOnlyHead
from src.modules.obs_projector import ObsProjector

from .lora_base import LoRABaseAgent


class LoRASpeakerAgent(LoRABaseAgent):
    """Speaker with LoRA-enabled backbone.

    Gradients from the listener's policy loss flow through the differentiable
    message, through the comm channel, through the projector, and into the
    speaker's LoRA weights — enabling cross-agent LLM adaptation.
    """

    def __init__(
        self,
        backbone: LoRALLM,
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
        z = self.encode_obs(obs, training=False)
        z_batch = z.unsqueeze(0)
        message = self.comm(z_batch).squeeze(0)
        value = self.value_head(z_batch).squeeze(0)
        env_action = np.zeros(self.env_action_dim, dtype=np.float32)
        return {
            "env_action": env_action,
            "message": message,
            "value": value,
            "z": z,
        }

    def act_batch(self, obs_list: list[np.ndarray], training: bool = True) -> dict[str, torch.Tensor]:
        """Batched forward. training=True enables LoRA gradient flow."""
        z_batch = self.encode_obs_batch(obs_list, training=training)
        messages = self.comm(z_batch)
        values = self.value_head(z_batch)
        return {
            "message": messages,
            "value": values,
            "z": z_batch,
        }
