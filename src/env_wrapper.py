from __future__ import annotations

import numpy as np
from pettingzoo.mpe import simple_speaker_listener_v4


class SpeakerListenerEnv:
    """Wraps PettingZoo simple_speaker_listener parallel env.

    Splits observations by agent and bypasses the env's built-in
    communication channel (we use our own differentiable channel).
    """

    SPEAKER = "speaker_0"
    LISTENER = "listener_0"

    def __init__(
        self,
        max_cycles: int = 25,
        continuous_actions: bool = True,
    ):
        self.env = simple_speaker_listener_v4.parallel_env(
            max_cycles=max_cycles,
            continuous_actions=continuous_actions,
        )
        self.max_cycles = max_cycles
        self._step_count = 0

    @property
    def speaker_obs_dim(self) -> int:
        return self.env.observation_space(self.SPEAKER).shape[0]

    @property
    def listener_obs_dim(self) -> int:
        return self.env.observation_space(self.LISTENER).shape[0]

    @property
    def speaker_action_dim(self) -> int:
        space = self.env.action_space(self.SPEAKER)
        return space.shape[0] if hasattr(space, "shape") else space.n

    @property
    def listener_action_dim(self) -> int:
        space = self.env.action_space(self.LISTENER)
        return space.shape[0] if hasattr(space, "shape") else space.n

    def reset(self, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Returns (speaker_obs, listener_obs)."""
        obs, _ = self.env.reset(seed=seed)
        self._step_count = 0
        return obs[self.SPEAKER], obs[self.LISTENER]

    def step(
        self,
        speaker_action: np.ndarray,
        listener_action: np.ndarray,
    ) -> tuple[np.ndarray | None, np.ndarray | None, float, bool, dict]:
        """Step env. Returns (speaker_obs, listener_obs, reward, done, info)."""
        actions = {
            self.SPEAKER: speaker_action,
            self.LISTENER: listener_action,
        }
        obs, rewards, terms, truncs, infos = self.env.step(actions)
        self._step_count += 1

        reward = rewards.get(self.LISTENER, 0.0)
        done = (
            terms.get(self.LISTENER, False)
            or truncs.get(self.LISTENER, False)
            or not self.env.agents
        )

        s_obs = obs.get(self.SPEAKER)
        l_obs = obs.get(self.LISTENER)
        return s_obs, l_obs, reward, done, infos

    def close(self):
        self.env.close()
