#!/usr/bin/env python3
"""Run all 4 communication methods and compare results."""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.agents.listener import ListenerAgent
from src.agents.speaker import SpeakerAgent
from src.backbone.llm import FrozenLLM
from src.comm import build_comm_channel
from src.config import CommConfig, Config, load_config
from src.env_wrapper import SpeakerListenerEnv
from src.modules.action_head import ContinuousActionHead, ValueOnlyHead
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter
from src.training.ppo import PPOTrainer
from src.utils.seeding import set_all_seeds
from src.utils.text_prompt import listener_obs_to_text, speaker_obs_to_text


def build_system(cfg: Config, backbone_speaker: FrozenLLM, backbone_listener: FrozenLLM, device: str):
    """Build speaker + listener + trainer from config, reusing frozen backbones."""
    comm_channel = build_comm_channel(cfg.comm, input_dim=cfg.modules.projector_hidden)
    msg_dim = comm_channel.message_dim()

    speaker = SpeakerAgent(
        backbone=backbone_speaker,
        obs_projector=ObsProjector(
            backbone_speaker.hidden_size, cfg.modules.projector_hidden, cfg.modules.projector_hidden,
        ),
        comm_channel=comm_channel,
        value_head=ValueOnlyHead(cfg.modules.projector_hidden, cfg.modules.action_hidden),
        obs_to_text_fn=speaker_obs_to_text,
        env_action_dim=3,
    ).to(device)

    listener = ListenerAgent(
        backbone=backbone_listener,
        obs_projector=ObsProjector(
            backbone_listener.hidden_size, cfg.modules.projector_hidden, cfg.modules.projector_hidden,
        ),
        receiver_adapter=ReceiverAdapter(
            hidden_dim=cfg.modules.projector_hidden,
            message_dim=msg_dim,
            output_dim=cfg.modules.adapter_hidden,
            adapter_hidden=cfg.modules.adapter_hidden,
        ),
        action_head=ContinuousActionHead(
            cfg.modules.adapter_hidden, action_dim=5, hidden_dim=cfg.modules.action_hidden,
        ),
        obs_to_text_fn=listener_obs_to_text,
    ).to(device)

    trainer = PPOTrainer(speaker, listener, cfg.training, device=device)
    return speaker, listener, trainer


def evaluate(speaker, listener, env, num_episodes=50):
    speaker.eval()
    listener.eval()
    rewards = []
    for ep in range(num_episodes):
        s_obs, l_obs = env.reset(seed=10000 + ep)
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                s_out = speaker.act(s_obs)
                l_out = listener.act(l_obs, s_out["message"], deterministic=True)
            s_obs, l_obs, reward, done, _ = env.step(s_out["env_action"], l_out["env_action"])
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)
    speaker.clear_cache()
    listener.clear_cache()
    return np.mean(rewards), np.std(rewards)


def run_method(
    method_name: str,
    cfg: Config,
    backbone_speaker: FrozenLLM,
    backbone_listener: FrozenLLM,
    device: str,
    total_episodes: int,
    eval_every: int,
    eval_episodes: int,
):
    """Train one method and return learning curve."""
    print(f"\n{'='*60}")
    print(f"  METHOD: {method_name}")
    print(f"  comm.type={cfg.comm.type}, comm.dim={cfg.comm.dim}")
    print(f"  total_episodes={total_episodes}")
    print(f"{'='*60}")

    set_all_seeds(cfg.seed)
    speaker, listener, trainer = build_system(cfg, backbone_speaker, backbone_listener, device)

    s_params = sum(p.numel() for p in speaker.parameters() if p.requires_grad)
    l_params = sum(p.numel() for p in listener.parameters() if p.requires_grad)
    print(f"  Trainable params: speaker={s_params:,}, listener={l_params:,}")

    # Checkpoint dir for this method
    ckpt_dir = Path("checkpoints") / method_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    env = SpeakerListenerEnv(max_cycles=cfg.env.max_cycles, continuous_actions=True)
    rollout_eps = cfg.training.rollout_episodes

    curve = {"episodes": [], "train_reward": [], "eval_reward": [], "eval_std": [], "policy_loss": []}
    episode_count = 0
    t_start = time.time()

    while episode_count < total_episodes:
        buffer, rm = trainer.collect_rollouts(env, rollout_eps)
        um = trainer.update(buffer)
        episode_count += rollout_eps

        curve["episodes"].append(episode_count)
        curve["train_reward"].append(rm["episode_reward_mean"])
        curve["policy_loss"].append(um["policy_loss"])

        if episode_count % eval_every < rollout_eps:
            eval_mean, eval_std = evaluate(speaker, listener, env, eval_episodes)
            curve["eval_reward"].append(eval_mean)
            curve["eval_std"].append(eval_std)
            elapsed = time.time() - t_start
            remaining = elapsed / episode_count * (total_episodes - episode_count)
            print(
                f"  [{episode_count:>6}/{total_episodes}] "
                f"train={rm['episode_reward_mean']:>8.2f}  "
                f"eval={eval_mean:>8.2f} +/- {eval_std:.2f}  "
                f"loss={um['policy_loss']:.4f}  "
                f"elapsed={elapsed/60:.1f}m  ETA={remaining/60:.1f}m"
            )
        else:
            curve["eval_reward"].append(None)
            curve["eval_std"].append(None)

    # Final eval
    eval_mean, eval_std = evaluate(speaker, listener, env, eval_episodes)
    elapsed = time.time() - t_start
    print(f"  FINAL: eval={eval_mean:.2f} +/- {eval_std:.2f}, time={elapsed/60:.1f}min")

    # Save final checkpoint
    ckpt_path = str(ckpt_dir / "checkpoint_final.pt")
    trainer.save_checkpoint(ckpt_path, episode_count)
    print(f"  Checkpoint saved to {ckpt_path}")

    env.close()
    return {
        "method": method_name,
        "final_eval_reward": eval_mean,
        "final_eval_std": eval_std,
        "time_minutes": elapsed / 60,
        "curve": curve,
        "checkpoint": ckpt_path,
    }


def print_summary(results: list[dict]):
    print(f"\n{'='*60}")
    print(f"  COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Method':<25} {'Eval Reward':>14} {'Time (min)':>12}")
    print(f"{'-'*25} {'-'*14} {'-'*12}")

    sorted_results = sorted(results, key=lambda r: r["final_eval_reward"], reverse=True)
    for r in sorted_results:
        reward_str = f"{r['final_eval_reward']:.2f} +/- {r['final_eval_std']:.2f}"
        print(f"{r['method']:<25} {reward_str:>14} {r['time_minutes']:>10.1f}")

    best = sorted_results[0]
    no_comm = next((r for r in results if "no_comm" in r["method"]), None)

    print(f"\nBest method: {best['method']} (reward={best['final_eval_reward']:.2f})")
    if no_comm:
        improvement = best["final_eval_reward"] - no_comm["final_eval_reward"]
        print(f"Improvement over no-comm: {improvement:+.2f}")


def main():
    parser = argparse.ArgumentParser(description="Compare all communication methods")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=5000, help="Episodes per method")
    parser.add_argument("--eval-every", type=int, default=500, help="Eval interval (episodes)")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comm-dim", type=int, default=8, help="SSR/continuous dim")
    parser.add_argument("--num-symbols", type=int, default=8, help="Discrete symbols")
    parser.add_argument("--output", type=str, default="results/comparison.json")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load LLM backbones ONCE (shared across all methods)
    print("Loading LLM backbone (shared across all methods)...")
    backbone = FrozenLLM("HuggingFaceTB/SmolLM2-135M-Instruct", device, "float16")
    print(f"  hidden_size={backbone.hidden_size}")

    methods = {
        "ssr": CommConfig(type="ssr", dim=args.comm_dim),
        "discrete": CommConfig(type="discrete", dim=args.comm_dim, num_symbols=args.num_symbols),
        "continuous": CommConfig(type="continuous", dim=args.comm_dim),
        "no_comm": CommConfig(type="none", dim=1),
    }

    results = []
    total_start = time.time()

    for method_name, comm_cfg in methods.items():
        cfg = load_config(args.config)
        cfg.seed = args.seed
        cfg.comm = comm_cfg

        result = run_method(
            method_name=method_name,
            cfg=cfg,
            backbone_speaker=backbone,
            backbone_listener=backbone,
            device=device,
            total_episodes=args.episodes,
            eval_every=args.eval_every,
            eval_episodes=args.eval_episodes,
        )
        results.append(result)

    total_time = (time.time() - total_start) / 60
    print(f"\nTotal time: {total_time:.1f} minutes")

    print_summary(results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert curve data for JSON serialization
    save_data = []
    for r in results:
        save_r = {k: v for k, v in r.items() if k != "curve"}
        save_r["curve"] = {
            k: [float(x) if x is not None else None for x in v]
            for k, v in r["curve"].items()
        }
        save_data.append(save_r)

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
