from __future__ import annotations

import numpy as np
import torch

from src.backbone.lora_llm import LoRALLM
from src.modules.action_head import ContinuousActionHead
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter

from .lora_base import LoRABaseAgent


class LoRAListenerAgent(LoRABaseAgent):
    """Listener with LoRA-enabled backbone.

    When LoRA is enabled on the listener, the policy loss gradients flow
    directly into the listener's LLM weights through the projector.
    Combined with a LoRA speaker, this creates bidirectional gradient flow
    through the communication channel.
    """

    def __init__(
        self,
        backbone: LoRALLM,
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
        z = self.encode_obs(obs, training=False)
        z_batch = z.unsqueeze(0)
        msg_batch = message.unsqueeze(0)
        h_tilde = self.adapter(z_batch, msg_batch)
        out = self.action_head.get_action(h_tilde, deterministic=deterministic)
        return {
            "env_action": out["action"].squeeze(0).detach().cpu().numpy(),
            "action": out["action"].squeeze(0),
            "raw_action": out["mean"].squeeze(0),
            "log_prob": out["log_prob"].squeeze(0),
            "value": out["value"].squeeze(0),
            "entropy": out["entropy"].squeeze(0),
            "z": z,
        }

    def act_batch(
        self,
        obs_list: list[np.ndarray],
        messages: torch.Tensor,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        z_batch = self.encode_obs_batch(obs_list, training=training)
        h_tilde = self.adapter(z_batch, messages)
        return self.action_head.get_action(h_tilde)

    def evaluate_batch(
        self,
        obs_list: list[np.ndarray],
        messages: torch.Tensor,
        raw_actions: torch.Tensor,
        training: bool = True,
    ) -> dict[str, torch.Tensor]:
        """Re-evaluate with gradient tracking through LoRA backbone."""
        z_batch = self.encode_obs_batch(obs_list, training=training)
        h_tilde = self.adapter(z_batch, messages)
        return self.action_head.evaluate_action(h_tilde, raw_actions)
