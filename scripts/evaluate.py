#!/usr/bin/env python3
"""Evaluate a trained checkpoint."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from src.agents.listener import ListenerAgent
from src.agents.speaker import SpeakerAgent
from src.backbone.llm import FrozenLLM
from src.comm import build_comm_channel
from src.config import load_config
from src.env_wrapper import SpeakerListenerEnv
from src.modules.action_head import ContinuousActionHead, ValueOnlyHead
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter
from src.utils.text_prompt import listener_obs_to_text, speaker_obs_to_text


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained agents")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--overrides", nargs="*", default=[])
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint .pt")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = args.device or cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Build agents (same as train.py)
    from scripts.train import build_agents
    speaker, listener = build_agents(cfg, device)

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    speaker.load_state_dict(ckpt["speaker_state_dict"])
    listener.load_state_dict(ckpt["listener_state_dict"])
    print(f"Loaded checkpoint from episode {ckpt['episode']}")

    speaker.eval()
    listener.eval()

    env = SpeakerListenerEnv(
        max_cycles=cfg.env.max_cycles,
        continuous_actions=cfg.env.continuous_actions,
    )

    rewards = []
    lengths = []
    messages_all = []

    for ep in range(args.episodes):
        s_obs, l_obs = env.reset(seed=ep)
        done = False
        ep_reward = 0.0
        ep_len = 0

        while not done:
            with torch.no_grad():
                s_out = speaker.act(s_obs)
                l_out = listener.act(l_obs, s_out["message"], deterministic=True)

            s_obs, l_obs, reward, done, _ = env.step(
                s_out["env_action"], l_out["env_action"]
            )
            ep_reward += reward
            ep_len += 1
            messages_all.append(s_out["message"].cpu().numpy())

            if done:
                break

        rewards.append(ep_reward)
        lengths.append(ep_len)

    speaker.clear_cache()
    listener.clear_cache()

    print(f"\nEvaluation over {args.episodes} episodes:")
    print(f"  Reward: {np.mean(rewards):.2f} +/- {np.std(rewards):.2f}")
    print(f"  Length: {np.mean(lengths):.1f} +/- {np.std(lengths):.1f}")

    # Message analysis
    messages = np.stack(messages_all)
    print(f"\nMessage statistics (dim={messages.shape[1]}):")
    print(f"  Mean: {messages.mean(axis=0)}")
    print(f"  Std:  {messages.std(axis=0)}")
    print(f"  Variance per dim: {messages.var(axis=0)}")

    env.close()


if __name__ == "__main__":
    main()
