"""Tests for training components."""

import numpy as np
import torch
from src.training.rollout_buffer import RolloutBuffer, Transition, _compute_gae


def test_gae():
    rewards = [1.0, 2.0, 3.0]
    values = [0.5, 1.0, 1.5]
    dones = [False, False, True]
    advantages = _compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    assert len(advantages) == 3
    # Last step done, so advantage = reward - value = 3.0 - 1.5 = 1.5
    assert abs(advantages[-1] - 1.5) < 1e-5


def test_rollout_buffer():
    buffer = RolloutBuffer()
    episode = []
    for i in range(5):
        t = Transition(
            speaker_obs=np.zeros(3),
            listener_obs=np.zeros(11),
            message=torch.zeros(8),
            raw_action=torch.randn(5),
            log_prob=-1.0,
            listener_value=0.5,
            speaker_value=0.3,
            reward=-1.0 + i * 0.1,
            done=(i == 4),
        )
        episode.append(t)
    buffer.add_episode(episode)
    assert buffer.total_steps == 5

    data = buffer.compute_returns_and_advantages(gamma=0.99, gae_lambda=0.95)
    assert data["raw_actions"].shape == (5, 5)
    assert data["old_log_probs"].shape == (5,)
    assert data["returns"].shape == (5,)
    assert data["advantages"].shape == (5,)
    assert len(data["speaker_obs"]) == 5


if __name__ == "__main__":
    test_gae()
    test_rollout_buffer()
    print("All training tests passed!")
