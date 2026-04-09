from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch


@dataclass
class Transition:
    speaker_obs: np.ndarray
    listener_obs: np.ndarray
    message: torch.Tensor        # detached, from speaker's comm channel
    raw_action: torch.Tensor     # pre-sigmoid action for re-evaluation
    log_prob: float              # listener's old log_prob
    listener_value: float
    speaker_value: float
    reward: float
    done: bool


class RolloutBuffer:
    """Stores transitions across episodes for PPO training."""

    def __init__(self):
        self.episodes: list[list[Transition]] = []

    def add_episode(self, transitions: list[Transition]):
        self.episodes.append(transitions)

    @property
    def total_steps(self) -> int:
        return sum(len(ep) for ep in self.episodes)

    def compute_returns_and_advantages(
        self,
        gamma: float,
        gae_lambda: float,
    ) -> dict[str, list]:
        """Flatten episodes and compute GAE advantages + returns."""
        all_speaker_obs = []
        all_listener_obs = []
        all_raw_actions = []
        all_old_log_probs = []
        all_returns = []
        all_advantages = []
        all_listener_values = []
        all_speaker_values = []

        for episode in self.episodes:
            rewards = [t.reward for t in episode]
            listener_values = [t.listener_value for t in episode]
            speaker_values = [t.speaker_value for t in episode]
            dones = [t.done for t in episode]

            # GAE for listener
            listener_advantages = _compute_gae(
                rewards, listener_values, dones, gamma, gae_lambda
            )
            listener_returns = [
                adv + val for adv, val in zip(listener_advantages, listener_values)
            ]

            # GAE for speaker (same rewards, different value baseline)
            speaker_advantages = _compute_gae(
                rewards, speaker_values, dones, gamma, gae_lambda
            )

            for i, t in enumerate(episode):
                all_speaker_obs.append(t.speaker_obs)
                all_listener_obs.append(t.listener_obs)
                all_raw_actions.append(t.raw_action)
                all_old_log_probs.append(t.log_prob)
                all_returns.append(listener_returns[i])
                all_advantages.append(listener_advantages[i])
                all_listener_values.append(t.listener_value)
                all_speaker_values.append(t.speaker_value)

        advantages = np.array(all_advantages, dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return {
            "speaker_obs": all_speaker_obs,
            "listener_obs": all_listener_obs,
            "raw_actions": torch.stack(all_raw_actions),
            "old_log_probs": torch.tensor(all_old_log_probs, dtype=torch.float32),
            "returns": torch.tensor(all_returns, dtype=torch.float32),
            "advantages": torch.tensor(advantages, dtype=torch.float32),
        }

    def clear(self):
        self.episodes.clear()


def _compute_gae(
    rewards: list[float],
    values: list[float],
    dones: list[bool],
    gamma: float,
    gae_lambda: float,
) -> list[float]:
    """Generalized Advantage Estimation."""
    advantages = []
    gae = 0.0
    n = len(rewards)
    for t in reversed(range(n)):
        if t == n - 1 or dones[t]:
            next_value = 0.0
        else:
            next_value = values[t + 1]
        delta = rewards[t] + gamma * next_value * (1.0 - float(dones[t])) - values[t]
        gae = delta + gamma * gae_lambda * (1.0 - float(dones[t])) * gae
        advantages.insert(0, gae)
    return advantages
