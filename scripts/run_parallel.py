#!/usr/bin/env python3
"""Run individual method+seed experiments and save results independently.

Usage:
    python scripts/run_parallel.py --method ssr --seed 42 --episodes 10000

Results saved to results/individual/{method}_s{seed}.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from src.backbone.llm import FrozenLLM
from src.comm import build_comm_channel
from src.config import CommConfig, Config, load_config
from src.env_wrapper import SpeakerListenerEnv
from src.agents.speaker import SpeakerAgent
from src.agents.listener import ListenerAgent
from src.modules.action_head import ContinuousActionHead, ValueOnlyHead
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter
from src.training.ppo import PPOTrainer
from src.utils.seeding import set_all_seeds
from src.utils.text_prompt import listener_obs_to_text, speaker_obs_to_text
from src.comm.vq_ssr import VQSSRChannel


METHODS = {
    "ssr": CommConfig(type="ssr", dim=8, normalize=True),
    "ssr_no_ln": CommConfig(type="ssr", dim=8, normalize=False),
    "ssr_v2": CommConfig(type="ssr_v2", dim=8, normalize=False, residual=True),
    "vq_ssr": CommConfig(type="vq_ssr", dim=8, num_codes=16),
    "discrete": CommConfig(type="discrete", dim=8, num_symbols=8),
    "continuous": CommConfig(type="continuous", dim=8),
    "no_comm": CommConfig(type="none", dim=1),
}


def build_system(cfg, backbone, device):
    comm_channel = build_comm_channel(cfg.comm, input_dim=cfg.modules.projector_hidden)
    msg_dim = comm_channel.message_dim()

    speaker = SpeakerAgent(
        backbone=backbone, obs_projector=ObsProjector(
            backbone.hidden_size, cfg.modules.projector_hidden, cfg.modules.projector_hidden),
        comm_channel=comm_channel,
        value_head=ValueOnlyHead(cfg.modules.projector_hidden, cfg.modules.action_hidden),
        obs_to_text_fn=speaker_obs_to_text, env_action_dim=3,
    ).to(device)

    listener = ListenerAgent(
        backbone=backbone, obs_projector=ObsProjector(
            backbone.hidden_size, cfg.modules.projector_hidden, cfg.modules.projector_hidden),
        receiver_adapter=ReceiverAdapter(
            hidden_dim=cfg.modules.projector_hidden, message_dim=msg_dim,
            output_dim=cfg.modules.adapter_hidden, adapter_hidden=cfg.modules.adapter_hidden),
        action_head=ContinuousActionHead(
            cfg.modules.adapter_hidden, action_dim=5, hidden_dim=cfg.modules.action_hidden),
        obs_to_text_fn=listener_obs_to_text,
    ).to(device)

    trainer = PPOTrainer(speaker, listener, cfg.training, device=device)
    return speaker, listener, trainer


def evaluate_with_analysis(speaker, listener, env, num_episodes=50):
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

    msgs = torch.stack(all_messages)
    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_reward_std": float(np.std(rewards)),
        "message_stats": {
            "per_dim_var": msgs.var(dim=0).tolist(),
            "per_dim_mean": msgs.mean(dim=0).tolist(),
            "overall_var": float(msgs.var()),
            "norm_mean": float(msgs.norm(dim=-1).mean()),
            "norm_std": float(msgs.norm(dim=-1).std()),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True, choices=list(METHODS.keys()))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--model", default="HuggingFaceTB/SmolLM2-135M-Instruct")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    comm_cfg = METHODS[args.method]
    cfg = load_config(args.config)
    cfg.comm = comm_cfg
    cfg.seed = args.seed
    set_all_seeds(args.seed)

    print(f"Method: {args.method}, Seed: {args.seed}, Episodes: {args.episodes}")

    backbone = FrozenLLM(args.model, device, "float16")
    speaker, listener, trainer = build_system(cfg, backbone, device)

    s_params = sum(p.numel() for p in speaker.parameters() if p.requires_grad)
    l_params = sum(p.numel() for p in listener.parameters() if p.requires_grad)
    print(f"Trainable: speaker={s_params:,}, listener={l_params:,}")

    env = SpeakerListenerEnv(max_cycles=cfg.env.max_cycles, continuous_actions=True)
    rollout_eps = cfg.training.rollout_episodes

    curve = {"episodes": [], "train_reward": [], "eval_reward": [], "eval_std": [],
             "message_var": [], "message_norm": []}
    episode_count = 0
    t_start = time.time()

    # Checkpoint and flag dirs
    ckpt_dir = Path(f"checkpoints/{args.method}_s{args.seed}")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    flag_dir = Path("results/flags")
    flag_dir.mkdir(parents=True, exist_ok=True)

    # Write training flag
    flag_path = flag_dir / f"{args.method}_s{args.seed}.running"
    flag_path.write_text(json.dumps({
        "method": args.method, "seed": args.seed, "episodes": args.episodes,
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"), "pid": os.getpid(),
    }))

    while episode_count < args.episodes:
        buffer, rm = trainer.collect_rollouts(env, rollout_eps)
        um = trainer.update(buffer)
        episode_count += rollout_eps

        curve["episodes"].append(float(episode_count))
        curve["train_reward"].append(float(rm["episode_reward_mean"]))
        curve["message_var"].append(float(um.get("message_var", 0)))
        curve["message_norm"].append(float(um.get("message_norm", 0)))

        if episode_count % args.eval_every < rollout_eps:
            ev = evaluate_with_analysis(speaker, listener, env, args.eval_episodes)
            curve["eval_reward"].append(ev["eval_reward_mean"])
            curve["eval_std"].append(ev["eval_reward_std"])
            elapsed = time.time() - t_start
            eta = elapsed / episode_count * (args.episodes - episode_count)
            print(f"  [{episode_count:>6}/{args.episodes}] eval={ev['eval_reward_mean']:>8.2f} +/- {ev['eval_reward_std']:.2f}  "
                  f"elapsed={elapsed/60:.1f}m  ETA={eta/60:.1f}m")
        else:
            curve["eval_reward"].append(None)
            curve["eval_std"].append(None)

        if cfg.comm.type == "discrete":
            progress = episode_count / args.episodes
            speaker.comm.tau = max(0.1, 1.0 * (1 - progress) + 0.1 * progress)

        # Save checkpoint every 2500 episodes
        if episode_count % 2500 < rollout_eps:
            ckpt_path = ckpt_dir / f"checkpoint_{episode_count}.pt"
            trainer.save_checkpoint(str(ckpt_path), episode_count)

    final = evaluate_with_analysis(speaker, listener, env, args.eval_episodes)
    elapsed = time.time() - t_start
    print(f"FINAL: eval={final['eval_reward_mean']:.2f} +/- {final['eval_reward_std']:.2f}, time={elapsed/60:.1f}min")

    env.close()

    # Save
    out_dir = Path("results/individual")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.method}_s{args.seed}.json"

    result = {
        "method": args.method,
        "seed": args.seed,
        "final_eval_reward": final["eval_reward_mean"],
        "final_eval_std": final["eval_reward_std"],
        "message_stats": final["message_stats"],
        "time_minutes": elapsed / 60,
        "curve": {k: [float(x) if x is not None else None for x in v] for k, v in curve.items()},
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}")

    # Save final checkpoint
    trainer.save_checkpoint(str(ckpt_dir / "checkpoint_final.pt"), episode_count)

    # Update flag: done
    flag_path.unlink(missing_ok=True)
    done_flag = flag_dir / f"{args.method}_s{args.seed}.done"
    done_flag.write_text(json.dumps({
        "method": args.method, "seed": args.seed,
        "final_reward": final["eval_reward_mean"],
        "time_minutes": elapsed / 60,
        "end_time": time.strftime("%Y-%m-%d %H:%M:%S"),
    }))


if __name__ == "__main__":
    main()
