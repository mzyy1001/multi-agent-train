#!/usr/bin/env python3
"""Main training entry point for multi-agent SSR communication experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

from src.agents.listener import ListenerAgent
from src.agents.speaker import SpeakerAgent
from src.backbone.llm import FrozenLLM
from src.comm import build_comm_channel
from src.config import load_config
from src.env_wrapper import SpeakerListenerEnv
from src.modules.action_head import ContinuousActionHead, ValueOnlyHead
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter
from src.training.ppo import PPOTrainer
from src.utils.logging import Logger
from src.utils.seeding import set_all_seeds
from src.utils.text_prompt import listener_obs_to_text, speaker_obs_to_text


def build_agents(cfg, device: str):
    """Build speaker and listener agents from config."""
    # Load frozen LLM backbones
    print(f"Loading speaker backbone: {cfg.speaker.model_id}")
    speaker_backbone = FrozenLLM(cfg.speaker.model_id, device, cfg.speaker.dtype)
    print(f"  hidden_size = {speaker_backbone.hidden_size}")

    print(f"Loading listener backbone: {cfg.listener.model_id}")
    listener_backbone = FrozenLLM(cfg.listener.model_id, device, cfg.listener.dtype)
    print(f"  hidden_size = {listener_backbone.hidden_size}")

    # Build communication channel
    comm_channel = build_comm_channel(cfg.comm, input_dim=cfg.modules.projector_hidden)
    msg_dim = comm_channel.message_dim()
    print(f"Comm channel: {cfg.comm.type}, message_dim={msg_dim}")

    # Build speaker
    speaker = SpeakerAgent(
        backbone=speaker_backbone,
        obs_projector=ObsProjector(
            speaker_backbone.hidden_size,
            cfg.modules.projector_hidden,
            cfg.modules.projector_hidden,
        ),
        comm_channel=comm_channel,
        value_head=ValueOnlyHead(cfg.modules.projector_hidden, cfg.modules.action_hidden),
        obs_to_text_fn=speaker_obs_to_text,
        env_action_dim=3,
    ).to(device)

    # Build listener
    listener = ListenerAgent(
        backbone=listener_backbone,
        obs_projector=ObsProjector(
            listener_backbone.hidden_size,
            cfg.modules.projector_hidden,
            cfg.modules.projector_hidden,
        ),
        receiver_adapter=ReceiverAdapter(
            hidden_dim=cfg.modules.projector_hidden,
            message_dim=msg_dim,
            output_dim=cfg.modules.adapter_hidden,
            adapter_hidden=cfg.modules.adapter_hidden,
        ),
        action_head=ContinuousActionHead(
            cfg.modules.adapter_hidden,
            action_dim=5,
            hidden_dim=cfg.modules.action_hidden,
        ),
        obs_to_text_fn=listener_obs_to_text,
    ).to(device)

    # Report trainable params
    s_params = sum(p.numel() for p in speaker.parameters() if p.requires_grad)
    l_params = sum(p.numel() for p in listener.parameters() if p.requires_grad)
    print(f"Trainable params: speaker={s_params:,}, listener={l_params:,}, total={s_params + l_params:,}")

    return speaker, listener


def evaluate(
    speaker: SpeakerAgent,
    listener: ListenerAgent,
    env: SpeakerListenerEnv,
    num_episodes: int,
) -> dict[str, float]:
    """Run evaluation episodes."""
    speaker.eval()
    listener.eval()
    rewards = []

    for _ in range(num_episodes):
        s_obs, l_obs = env.reset()
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                speaker_out = speaker.act(s_obs)
                listener_out = listener.act(l_obs, speaker_out["message"], deterministic=True)
            s_obs, l_obs, reward, done, _ = env.step(
                speaker_out["env_action"], listener_out["env_action"]
            )
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)

    speaker.clear_cache()
    listener.clear_cache()

    return {
        "eval_reward_mean": np.mean(rewards),
        "eval_reward_std": np.std(rewards),
    }


def main():
    parser = argparse.ArgumentParser(description="Train multi-agent SSR communication")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument(
        "--overrides", nargs="*", default=[],
        help="Config overlay files, e.g. comm/ssr.yaml model/smol_qwen.yaml",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument(
        "--set", nargs="*", default=[], dest="cli_overrides",
        help="Key=value overrides, e.g. comm.dim=16 training.total_episodes=10000",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides, args.cli_overrides)
    if args.seed is not None:
        cfg.seed = args.seed
    if args.device is not None:
        cfg.device = args.device

    set_all_seeds(cfg.seed)
    device = cfg.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
        cfg.device = device

    run_name = args.run_name or f"{cfg.comm.type}_d{cfg.comm.dim}_s{cfg.seed}"
    logger = Logger(cfg.logging, run_name=run_name)
    print(f"Run: {run_name}")

    # Environment
    env = SpeakerListenerEnv(
        max_cycles=cfg.env.max_cycles,
        continuous_actions=cfg.env.continuous_actions,
    )

    # Agents
    speaker, listener = build_agents(cfg, device)

    # Trainer
    trainer = PPOTrainer(speaker, listener, cfg.training, device=device)

    # Resume if requested
    start_episode = 0
    if cfg.checkpoint.resume:
        start_episode = trainer.load_checkpoint(cfg.checkpoint.resume)
        print(f"Resumed from episode {start_episode}")

    # Checkpoint dir
    ckpt_dir = Path(cfg.checkpoint.dir) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    episode_count = start_episode
    total = cfg.training.total_episodes
    rollout_eps = cfg.training.rollout_episodes

    pbar = trange(
        episode_count, total, rollout_eps,
        desc="Training", unit="ep",
    )

    for _ in pbar:
        # Collect rollouts
        buffer, rollout_metrics = trainer.collect_rollouts(env, rollout_eps)
        episode_count += rollout_eps

        # PPO update
        update_metrics = trainer.update(buffer)

        # Log
        all_metrics = {**rollout_metrics, **update_metrics}

        if episode_count % cfg.logging.log_interval < rollout_eps:
            logger.log(all_metrics, step=episode_count, prefix="train/")
            pbar.set_postfix(
                reward=f"{rollout_metrics['episode_reward_mean']:.2f}",
                loss=f"{update_metrics['policy_loss']:.4f}",
            )

        # Evaluate
        if episode_count % cfg.logging.eval_interval < rollout_eps:
            eval_metrics = evaluate(
                speaker, listener, env, cfg.logging.eval_episodes
            )
            logger.log(eval_metrics, step=episode_count, prefix="eval/")
            print(
                f"\n[Eval @ {episode_count}] "
                f"reward={eval_metrics['eval_reward_mean']:.2f} "
                f"+/- {eval_metrics['eval_reward_std']:.2f}"
            )

        # Save checkpoint
        if episode_count % cfg.logging.save_interval < rollout_eps:
            ckpt_path = ckpt_dir / f"checkpoint_{episode_count}.pt"
            trainer.save_checkpoint(str(ckpt_path), episode_count)

        # Anneal Gumbel temperature for discrete baseline
        if cfg.comm.type == "discrete":
            progress = episode_count / total
            new_tau = max(
                cfg.comm.gumbel_tau_min,
                cfg.comm.gumbel_tau * (1 - progress) + cfg.comm.gumbel_tau_min * progress,
            )
            speaker.comm.tau = new_tau

    # Final eval
    eval_metrics = evaluate(speaker, listener, env, cfg.logging.eval_episodes)
    print(f"\nFinal eval: reward={eval_metrics['eval_reward_mean']:.2f}")
    logger.log(eval_metrics, step=episode_count, prefix="eval/")

    # Save final checkpoint
    trainer.save_checkpoint(str(ckpt_dir / "checkpoint_final.pt"), episode_count)

    logger.close()
    env.close()
    print("Training complete.")


if __name__ == "__main__":
    main()
