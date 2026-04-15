#!/usr/bin/env python3
"""Enhanced comparison: all comm methods, multi-seed, longer training, message analysis."""

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
from src.comm.vq_ssr import VQSSRChannel
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


def evaluate_with_analysis(speaker, listener, env, num_episodes=50):
    """Evaluate and collect message statistics."""
    speaker.eval()
    listener.eval()
    rewards = []
    all_messages = []

    for ep in range(num_episodes):
        s_obs, l_obs = env.reset(seed=10000 + ep)
        done = False
        ep_reward = 0.0
        while not done:
            with torch.no_grad():
                s_out = speaker.act(s_obs)
                l_out = listener.act(l_obs, s_out["message"], deterministic=True)
                all_messages.append(s_out["message"].cpu())
            s_obs, l_obs, reward, done, _ = env.step(s_out["env_action"], l_out["env_action"])
            ep_reward += reward
            if done:
                break
        rewards.append(ep_reward)

    speaker.clear_cache()
    listener.clear_cache()

    # Message analysis
    msgs = torch.stack(all_messages)  # (N, dim)
    msg_stats = {
        "per_dim_var": msgs.var(dim=0).tolist(),
        "per_dim_mean": msgs.mean(dim=0).tolist(),
        "overall_var": msgs.var().item(),
        "norm_mean": msgs.norm(dim=-1).mean().item(),
        "norm_std": msgs.norm(dim=-1).std().item(),
    }

    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "message_stats": msg_stats,
    }


def run_method(
    method_name: str,
    cfg: Config,
    backbone_speaker: FrozenLLM,
    backbone_listener: FrozenLLM,
    device: str,
    total_episodes: int,
    eval_every: int,
    eval_episodes: int,
    seed: int,
):
    """Train one method with one seed and return results."""
    run_label = f"{method_name}_s{seed}"
    print(f"\n{'='*60}")
    print(f"  METHOD: {run_label}")
    print(f"  comm.type={cfg.comm.type}, comm.dim={cfg.comm.dim}")
    print(f"  total_episodes={total_episodes}, seed={seed}")
    print(f"{'='*60}")

    cfg.seed = seed
    set_all_seeds(seed)
    speaker, listener, trainer = build_system(cfg, backbone_speaker, backbone_listener, device)

    s_params = sum(p.numel() for p in speaker.parameters() if p.requires_grad)
    l_params = sum(p.numel() for p in listener.parameters() if p.requires_grad)
    print(f"  Trainable params: speaker={s_params:,}, listener={l_params:,}")

    env = SpeakerListenerEnv(max_cycles=cfg.env.max_cycles, continuous_actions=True)
    rollout_eps = cfg.training.rollout_episodes

    curve = {"episodes": [], "train_reward": [], "eval_reward": [], "eval_std": [], "policy_loss": [],
             "message_var": [], "message_norm": []}
    episode_count = 0
    t_start = time.time()

    while episode_count < total_episodes:
        buffer, rm = trainer.collect_rollouts(env, rollout_eps)
        um = trainer.update(buffer)
        episode_count += rollout_eps

        curve["episodes"].append(float(episode_count))
        curve["train_reward"].append(float(rm["episode_reward_mean"]))
        curve["policy_loss"].append(float(um["policy_loss"]))
        curve["message_var"].append(float(um.get("message_var", 0)))
        curve["message_norm"].append(float(um.get("message_norm", 0)))

        if episode_count % eval_every < rollout_eps:
            eval_result = evaluate_with_analysis(speaker, listener, env, eval_episodes)
            curve["eval_reward"].append(eval_result["eval_reward_mean"])
            curve["eval_std"].append(eval_result["eval_reward_std"])
            elapsed = time.time() - t_start
            remaining = elapsed / episode_count * (total_episodes - episode_count)
            extra = ""
            if isinstance(speaker.comm, VQSSRChannel):
                extra = f"  perp={um.get('vq_perplexity', 0):.1f}"
            print(
                f"  [{episode_count:>6}/{total_episodes}] "
                f"train={rm['episode_reward_mean']:>8.2f}  "
                f"eval={eval_result['eval_reward_mean']:>8.2f} +/- {eval_result['eval_reward_std']:.2f}  "
                f"loss={um['policy_loss']:.4f}  msg_var={um.get('message_var', 0):.4f}"
                f"{extra}  "
                f"elapsed={elapsed/60:.1f}m  ETA={remaining/60:.1f}m"
            )
        else:
            curve["eval_reward"].append(None)
            curve["eval_std"].append(None)

        # Anneal Gumbel temperature for discrete baseline
        if cfg.comm.type == "discrete":
            progress = episode_count / total_episodes
            new_tau = max(
                cfg.comm.gumbel_tau_min,
                cfg.comm.gumbel_tau * (1 - progress) + cfg.comm.gumbel_tau_min * progress,
            )
            speaker.comm.tau = new_tau

    # Final eval with full message analysis
    final_eval = evaluate_with_analysis(speaker, listener, env, eval_episodes)
    elapsed = time.time() - t_start
    print(f"  FINAL: eval={final_eval['eval_reward_mean']:.2f} +/- {final_eval['eval_reward_std']:.2f}, time={elapsed/60:.1f}min")

    env.close()
    return {
        "method": method_name,
        "seed": seed,
        "final_eval_reward": final_eval["eval_reward_mean"],
        "final_eval_std": final_eval["eval_reward_std"],
        "message_stats": final_eval["message_stats"],
        "time_minutes": elapsed / 60,
        "curve": curve,
    }


def aggregate_seeds(results: list[dict]) -> dict:
    """Aggregate results across seeds for the same method."""
    rewards = [r["final_eval_reward"] for r in results]
    return {
        "method": results[0]["method"],
        "seeds": [r["seed"] for r in results],
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards)),
        "per_seed": results,
    }


def print_summary(aggregated: list[dict]):
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY (multi-seed)")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'Mean Reward':>14} {'Std (seeds)':>12} {'Range':>20}")
    print(f"{'-'*20} {'-'*14} {'-'*12} {'-'*20}")

    sorted_agg = sorted(aggregated, key=lambda r: r["mean_reward"], reverse=True)
    for r in sorted_agg:
        range_str = f"[{r['min_reward']:.1f}, {r['max_reward']:.1f}]"
        print(f"{r['method']:<20} {r['mean_reward']:>10.2f}     {r['std_reward']:>8.2f}     {range_str:>20}")

    best = sorted_agg[0]
    no_comm = next((r for r in aggregated if "no_comm" in r["method"]), None)

    print(f"\nBest method: {best['method']} (mean_reward={best['mean_reward']:.2f})")
    if no_comm:
        improvement = best["mean_reward"] - no_comm["mean_reward"]
        print(f"Improvement over no-comm: {improvement:+.2f}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced comparison: multi-seed, new methods")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=10000, help="Episodes per method per seed")
    parser.add_argument("--eval-every", type=int, default=500, help="Eval interval (episodes)")
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    parser.add_argument("--comm-dim", type=int, default=8, help="SSR/continuous dim")
    parser.add_argument("--num-symbols", type=int, default=8, help="Discrete symbols")
    parser.add_argument("--output", type=str, default="results/comparison_v2.json")
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Methods to run (default: all). Options: ssr, ssr_no_ln, ssr_v2, vq_ssr, discrete, continuous, no_comm")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load LLM backbones ONCE
    print("Loading LLM backbone (shared across all methods)...")
    backbone = FrozenLLM("HuggingFaceTB/SmolLM2-135M-Instruct", device, "float16")
    print(f"  hidden_size={backbone.hidden_size}")

    all_methods = {
        "ssr": CommConfig(type="ssr", dim=args.comm_dim, normalize=True),
        "ssr_no_ln": CommConfig(type="ssr", dim=args.comm_dim, normalize=False),
        "ssr_v2": CommConfig(type="ssr_v2", dim=args.comm_dim, normalize=False, residual=True),
        "vq_ssr": CommConfig(type="vq_ssr", dim=args.comm_dim, num_codes=16),
        "discrete": CommConfig(type="discrete", dim=args.comm_dim, num_symbols=args.num_symbols),
        "continuous": CommConfig(type="continuous", dim=args.comm_dim),
        "no_comm": CommConfig(type="none", dim=1),
    }

    selected = args.methods or list(all_methods.keys())
    methods = {k: v for k, v in all_methods.items() if k in selected}

    all_results = []
    per_method_results = defaultdict(list)
    total_start = time.time()

    for method_name, comm_cfg in methods.items():
        for seed in args.seeds:
            cfg = load_config(args.config)
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
                seed=seed,
            )
            all_results.append(result)
            per_method_results[method_name].append(result)

    total_time = (time.time() - total_start) / 60
    print(f"\nTotal time: {total_time:.1f} minutes")

    # Aggregate across seeds
    aggregated = [aggregate_seeds(results) for results in per_method_results.values()]
    print_summary(aggregated)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "config": {
            "episodes": args.episodes,
            "seeds": args.seeds,
            "comm_dim": args.comm_dim,
            "eval_episodes": args.eval_episodes,
        },
        "aggregated": aggregated,
        "total_time_minutes": total_time,
    }

    # Clean curve data for JSON serialization
    for agg in save_data["aggregated"]:
        for seed_result in agg["per_seed"]:
            seed_result["curve"] = {
                k: [float(x) if x is not None else None for x in v]
                for k, v in seed_result["curve"].items()
            }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
