"""Tests for environment wrapper."""

import numpy as np
from src.env_wrapper import SpeakerListenerEnv


def test_env_reset():
    env = SpeakerListenerEnv(max_cycles=10, continuous_actions=True)
    s_obs, l_obs = env.reset(seed=42)
    assert s_obs.shape == (3,), f"Speaker obs shape: {s_obs.shape}"
    assert l_obs.shape == (11,), f"Listener obs shape: {l_obs.shape}"
    env.close()


def test_env_step():
    env = SpeakerListenerEnv(max_cycles=10, continuous_actions=True)
    s_obs, l_obs = env.reset(seed=42)
    s_act = np.zeros(3, dtype=np.float32)
    l_act = np.zeros(5, dtype=np.float32)
    s_obs2, l_obs2, reward, done, info = env.step(s_act, l_act)
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    env.close()


def test_env_full_episode():
    env = SpeakerListenerEnv(max_cycles=5, continuous_actions=True)
    s_obs, l_obs = env.reset(seed=42)
    steps = 0
    while True:
        s_act = np.zeros(3, dtype=np.float32)
        l_act = np.random.randn(5).astype(np.float32).clip(0, 1)
        s_obs, l_obs, reward, done, _ = env.step(s_act, l_act)
        steps += 1
        if done:
            break
    assert steps <= 5
    env.close()


if __name__ == "__main__":
    test_env_reset()
    test_env_step()
    test_env_full_episode()
    print("All env tests passed!")
