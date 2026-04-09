from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.agents.listener import ListenerAgent
from src.agents.speaker import SpeakerAgent
from src.config import TrainingConfig
from src.env_wrapper import SpeakerListenerEnv

from .rollout_buffer import RolloutBuffer, Transition


class PPOTrainer:
    """PPO trainer for the two-agent system.

    The critical gradient path for communication:
      reward -> listener_policy_loss -> action_head -> receiver_adapter
             -> message (from speaker's comm channel)
             -> comm_channel -> speaker's obs_projector

    During the PPO update, both agents are re-forwarded with gradient
    tracking so that the speaker's trainable modules receive gradients
    through the listener's policy loss.
    """

    def __init__(
        self,
        speaker: SpeakerAgent,
        listener: ListenerAgent,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.speaker = speaker
        self.listener = listener
        self.config = config
        self.device = device

        # Single optimizer over ALL trainable params from both agents
        self.all_params = list(speaker.parameters()) + list(listener.parameters())
        self.optimizer = torch.optim.Adam(self.all_params, lr=config.lr)

    def collect_rollouts(
        self,
        env: SpeakerListenerEnv,
        num_episodes: int,
    ) -> RolloutBuffer:
        """Run episodes, collecting transitions. No gradient tracking."""
        buffer = RolloutBuffer()
        self.speaker.eval()
        self.listener.eval()
        # Clear cache at start of new rollout (stale from previous iteration)
        self.speaker.clear_cache()
        self.listener.clear_cache()

        episode_rewards = []

        for _ in range(num_episodes):
            s_obs, l_obs = env.reset()
            episode: list[Transition] = []
            done = False
            ep_reward = 0.0

            while not done:
                with torch.no_grad():
                    speaker_out = self.speaker.act(s_obs)
                    message = speaker_out["message"]

                    listener_out = self.listener.act(l_obs, message)

                s_obs_next, l_obs_next, reward, done, _ = env.step(
                    speaker_out["env_action"],
                    listener_out["env_action"],
                )

                episode.append(Transition(
                    speaker_obs=s_obs.copy(),
                    listener_obs=l_obs.copy(),
                    message=message.detach(),
                    raw_action=listener_out["raw_action"].detach(),
                    log_prob=listener_out["log_prob"].item(),
                    listener_value=listener_out["value"].item(),
                    speaker_value=speaker_out["value"].item(),
                    reward=reward,
                    done=done,
                ))
                ep_reward += reward

                if done:
                    break
                s_obs, l_obs = s_obs_next, l_obs_next

            buffer.add_episode(episode)
            episode_rewards.append(ep_reward)

        return buffer, {
            "episode_reward_mean": np.mean(episode_rewards),
            "episode_reward_std": np.std(episode_rewards),
            "episode_length_mean": buffer.total_steps / max(num_episodes, 1),
        }

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        """PPO update with re-forward through both agents for differentiable comm."""
        self.speaker.train()
        self.listener.train()

        data = buffer.compute_returns_and_advantages(
            self.config.gamma, self.config.gae_lambda
        )

        raw_actions = data["raw_actions"].to(self.device)
        old_log_probs = data["old_log_probs"].to(self.device)
        returns = data["returns"].to(self.device)
        advantages = data["advantages"].to(self.device)

        total_steps = len(data["speaker_obs"])
        metrics = defaultdict(float)
        update_count = 0

        for epoch in range(self.config.ppo_epochs):
            indices = np.random.permutation(total_steps)

            for start in range(0, total_steps, self.config.minibatch_size):
                end = min(start + self.config.minibatch_size, total_steps)
                mb_idx = indices[start:end]

                mb_speaker_obs = [data["speaker_obs"][i] for i in mb_idx]
                mb_listener_obs = [data["listener_obs"][i] for i in mb_idx]
                mb_raw_actions = raw_actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Re-forward speaker to get messages WITH gradients
                speaker_out = self.speaker.act_batch(mb_speaker_obs)
                messages = speaker_out["message"]  # (mb, comm_dim) - gradient tracked

                # Re-forward listener with gradient-tracked messages
                listener_out = self.listener.evaluate_batch(
                    mb_listener_obs, messages, mb_raw_actions
                )

                # PPO clipped objective
                new_log_probs = listener_out["log_prob"]
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = ratio.clamp(
                    1.0 - self.config.clip_eps,
                    1.0 + self.config.clip_eps,
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value losses
                listener_value_loss = F.mse_loss(
                    listener_out["value"], mb_returns
                )
                speaker_value_loss = F.mse_loss(
                    speaker_out["value"], mb_returns
                )

                # Entropy bonus
                entropy_loss = -listener_out["entropy"].mean()

                # Total loss
                loss = (
                    policy_loss
                    + self.config.value_coef * (listener_value_loss + speaker_value_loss)
                    + self.config.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(
                    self.all_params, self.config.max_grad_norm
                )
                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["listener_value_loss"] += listener_value_loss.item()
                metrics["speaker_value_loss"] += speaker_value_loss.item()
                metrics["entropy"] += -entropy_loss.item()
                metrics["grad_norm"] += grad_norm.item()
                metrics["approx_kl"] += (
                    (mb_old_log_probs - new_log_probs).mean().item()
                )
                update_count += 1

        return {k: v / max(update_count, 1) for k, v in metrics.items()}

    def save_checkpoint(self, path: str, episode: int):
        torch.save({
            "episode": episode,
            "speaker_state_dict": self.speaker.state_dict(),
            "listener_state_dict": self.listener.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }, path)

    def load_checkpoint(self, path: str) -> int:
        ckpt = torch.load(path, map_location=self.device)
        self.speaker.load_state_dict(ckpt["speaker_state_dict"])
        self.listener.load_state_dict(ckpt["listener_state_dict"])
        self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        return ckpt["episode"]
