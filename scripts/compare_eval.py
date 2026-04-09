#!/usr/bin/env python3
"""Train all 4 methods, then evaluate on the EXACT same episodes for fair comparison."""

from __future__ import annotations

import argparse
import json
import time
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


def build_system(cfg, backbone_speaker, backbone_listener, device):
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


def train_method(name, cfg, backbone, device, total_episodes):
    """Train one method, return trained speaker + listener."""
    print(f"\n  Training: {name} ({total_episodes} episodes)...")
    set_all_seeds(cfg.seed)
    speaker, listener, trainer = build_system(cfg, backbone, backbone, device)
    env = SpeakerListenerEnv(max_cycles=cfg.env.max_cycles, continuous_actions=True)
    rollout_eps = cfg.training.rollout_episodes

    ckpt_dir = Path("checkpoints") / name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    episode_count = 0
    t0 = time.time()
    while episode_count < total_episodes:
        buffer, rm = trainer.collect_rollouts(env, rollout_eps)
        trainer.update(buffer)
        episode_count += rollout_eps
        if episode_count % 1000 < rollout_eps:
            elapsed = time.time() - t0
            print(f"    [{episode_count:>5}/{total_episodes}] train_reward={rm['episode_reward_mean']:.2f}  ({elapsed/60:.1f}m)")

    # Save checkpoint
    ckpt_path = str(ckpt_dir / "checkpoint_final.pt")
    trainer.save_checkpoint(ckpt_path, episode_count)
    print(f"  Checkpoint saved to {ckpt_path}")

    env.close()
    elapsed = time.time() - t0
    print(f"  Done: {name} in {elapsed/60:.1f}min")
    return speaker, listener


def eval_on_same_episodes(
    models: dict[str, tuple[SpeakerAgent, ListenerAgent]],
    num_episodes: int,
    max_cycles: int,
    eval_seed_start: int = 99999,
):
    """Evaluate all models on the EXACT same episodes (same seeds)."""
    env = SpeakerListenerEnv(max_cycles=max_cycles, continuous_actions=True)

    # results[method][episode_idx] = {"reward": ..., "steps": ..., "final_dist": ...}
    results = {name: [] for name in models}

    print(f"\n{'='*70}")
    print(f"  EVALUATING ALL METHODS ON {num_episodes} IDENTICAL EPISODES")
    print(f"{'='*70}")

    for ep_idx in range(num_episodes):
        seed = eval_seed_start + ep_idx

        for name, (speaker, listener) in models.items():
            speaker.eval()
            listener.eval()

            # Reset with SAME seed for every method
            s_obs, l_obs = env.reset(seed=seed)
            done = False
            ep_reward = 0.0
            steps = 0

            while not done:
                with torch.no_grad():
                    s_out = speaker.act(s_obs)
                    l_out = listener.act(l_obs, s_out["message"], deterministic=True)
                s_obs_next, l_obs_next, reward, done, _ = env.step(
                    s_out["env_action"], l_out["env_action"]
                )
                ep_reward += reward
                steps += 1
                if done:
                    break
                s_obs, l_obs = s_obs_next, l_obs_next

            results[name].append({
                "seed": seed,
                "reward": ep_reward,
                "steps": steps,
            })

            speaker.clear_cache()
            listener.clear_cache()

    env.close()
    return results


def print_results(results: dict, num_episodes: int):
    """Print detailed comparison."""
    print(f"\n{'='*70}")
    print(f"  RESULTS: {num_episodes} identical episodes")
    print(f"{'='*70}")

    # Summary table
    summaries = {}
    for name, episodes in results.items():
        rewards = [e["reward"] for e in episodes]
        summaries[name] = {
            "mean": np.mean(rewards),
            "std": np.std(rewards),
            "median": np.median(rewards),
            "min": np.min(rewards),
            "max": np.max(rewards),
        }

    sorted_methods = sorted(summaries.keys(), key=lambda n: summaries[n]["mean"], reverse=True)

    print(f"\n{'Method':<18} {'Mean':>10} {'Std':>10} {'Median':>10} {'Min':>10} {'Max':>10}")
    print(f"{'-'*18} {'-'*10} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    for name in sorted_methods:
        s = summaries[name]
        print(f"{name:<18} {s['mean']:>10.2f} {s['std']:>10.2f} {s['median']:>10.2f} {s['min']:>10.2f} {s['max']:>10.2f}")

    # Head-to-head: for each episode, which method won?
    print(f"\n  HEAD-TO-HEAD (per-episode wins):")
    win_counts = {name: 0 for name in results}
    for ep_idx in range(num_episodes):
        best_name = max(results.keys(), key=lambda n: results[n][ep_idx]["reward"])
        win_counts[best_name] += 1
    for name in sorted_methods:
        pct = win_counts[name] / num_episodes * 100
        print(f"    {name:<18} {win_counts[name]:>4} wins ({pct:.1f}%)")

    # Pairwise: SSR vs each other method
    if "ssr" in results:
        print(f"\n  PAIRWISE vs SSR:")
        ssr_rewards = [e["reward"] for e in results["ssr"]]
        for name in sorted_methods:
            if name == "ssr":
                continue
            other_rewards = [e["reward"] for e in results[name]]
            ssr_better = sum(1 for s, o in zip(ssr_rewards, other_rewards) if s > o)
            ties = sum(1 for s, o in zip(ssr_rewards, other_rewards) if abs(s - o) < 1e-6)
            other_better = num_episodes - ssr_better - ties
            diff = np.mean(ssr_rewards) - np.mean(other_rewards)
            print(f"    SSR vs {name:<14} SSR wins {ssr_better}, loses {other_better}, ties {ties}  (mean diff: {diff:+.2f})")

    return summaries


def main():
    parser = argparse.ArgumentParser(description="Train all 4 methods, eval on same episodes")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--train-episodes", type=int, default=5000, help="Training episodes per method")
    parser.add_argument("--eval-episodes", type=int, default=100, help="Number of eval episodes")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--comm-dim", type=int, default=8)
    parser.add_argument("--output", type=str, default="results/eval_comparison.json")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("Loading LLM backbone...")
    backbone = FrozenLLM("HuggingFaceTB/SmolLM2-135M-Instruct", device, "float16")
    print(f"  hidden_size={backbone.hidden_size}")

    methods_cfg = {
        "ssr":        CommConfig(type="ssr", dim=args.comm_dim),
        "discrete":   CommConfig(type="discrete", dim=args.comm_dim, num_symbols=8),
        "continuous":  CommConfig(type="continuous", dim=args.comm_dim),
        "no_comm":    CommConfig(type="none", dim=1),
    }

    # Phase 1: Train all methods
    print(f"\n{'='*70}")
    print(f"  PHASE 1: TRAINING ({args.train_episodes} episodes each)")
    print(f"{'='*70}")

    trained_models = {}
    t_total = time.time()

    for name, comm_cfg in methods_cfg.items():
        cfg = load_config(args.config)
        cfg.seed = args.seed
        cfg.comm = comm_cfg
        speaker, listener = train_method(name, cfg, backbone, device, args.train_episodes)
        trained_models[name] = (speaker, listener)

    train_time = (time.time() - t_total) / 60
    print(f"\nAll training done in {train_time:.1f} min")

    # Phase 2: Evaluate all on same episodes
    print(f"\n{'='*70}")
    print(f"  PHASE 2: EVALUATION ({args.eval_episodes} identical episodes)")
    print(f"{'='*70}")

    t_eval = time.time()
    results = eval_on_same_episodes(
        trained_models,
        num_episodes=args.eval_episodes,
        max_cycles=25,
    )
    eval_time = (time.time() - t_eval) / 60
    print(f"Evaluation done in {eval_time:.1f} min")

    summaries = print_results(results, args.eval_episodes)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "config": {
            "train_episodes": args.train_episodes,
            "eval_episodes": args.eval_episodes,
            "seed": args.seed,
            "comm_dim": args.comm_dim,
        },
        "summaries": {k: {kk: float(vv) for kk, vv in v.items()} for k, v in summaries.items()},
        "per_episode": {
            name: [{"seed": e["seed"], "reward": float(e["reward"]), "steps": e["steps"]} for e in eps]
            for name, eps in results.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")
    print(f"Total time: {(time.time() - t_total) / 60:.1f} min")


if __name__ == "__main__":
    main()
