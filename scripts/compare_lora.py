#!/usr/bin/env python3
"""Compare frozen vs LoRA-speaker vs LoRA-both across communication methods.

This is the key experiment for the "gradient highway" thesis:
Does differentiable communication enable meaningful cross-agent LLM adaptation?

If yes: LoRA-speaker with differentiable comm (SSR/VQ-SSR) should outperform
        LoRA-speaker with discrete comm (where gradients can't reach the speaker's LLM).
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from src.backbone.llm import FrozenLLM
from src.backbone.lora_llm import LoRALLM
from src.agents.speaker import SpeakerAgent
from src.agents.listener import ListenerAgent
from src.agents.lora_speaker import LoRASpeakerAgent
from src.agents.lora_listener import LoRAListenerAgent
from src.comm import build_comm_channel
from src.config import CommConfig, Config, load_config
from src.env_wrapper import SpeakerListenerEnv
from src.modules.action_head import ContinuousActionHead, ValueOnlyHead
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter
from src.training.ppo import PPOTrainer
from src.training.ppo_lora import LoRAPPOTrainer
from src.utils.seeding import set_all_seeds
from src.utils.text_prompt import listener_obs_to_text, speaker_obs_to_text


def build_frozen_system(cfg, backbone, device):
    """Build frozen-backbone system (baseline)."""
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


def build_lora_system(cfg, model_id, device, lora_rank, lora_speaker=True, lora_listener=True):
    """Build LoRA-enabled system."""
    comm_channel = build_comm_channel(cfg.comm, input_dim=cfg.modules.projector_hidden)
    msg_dim = comm_channel.message_dim()

    # Build speaker
    if lora_speaker:
        spk_backbone = LoRALLM(model_id, device, "float16", lora_rank=lora_rank)
        speaker = LoRASpeakerAgent(
            backbone=spk_backbone, obs_projector=ObsProjector(
                spk_backbone.hidden_size, cfg.modules.projector_hidden, cfg.modules.projector_hidden),
            comm_channel=comm_channel,
            value_head=ValueOnlyHead(cfg.modules.projector_hidden, cfg.modules.action_hidden),
            obs_to_text_fn=speaker_obs_to_text, env_action_dim=3,
        ).to(device)
    else:
        spk_backbone = FrozenLLM(model_id, device, "float16")
        speaker = SpeakerAgent(
            backbone=spk_backbone, obs_projector=ObsProjector(
                spk_backbone.hidden_size, cfg.modules.projector_hidden, cfg.modules.projector_hidden),
            comm_channel=comm_channel,
            value_head=ValueOnlyHead(cfg.modules.projector_hidden, cfg.modules.action_hidden),
            obs_to_text_fn=speaker_obs_to_text, env_action_dim=3,
        ).to(device)

    # Build listener
    if lora_listener:
        lst_backbone = LoRALLM(model_id, device, "float16", lora_rank=lora_rank)
        listener = LoRAListenerAgent(
            backbone=lst_backbone, obs_projector=ObsProjector(
                lst_backbone.hidden_size, cfg.modules.projector_hidden, cfg.modules.projector_hidden),
            receiver_adapter=ReceiverAdapter(
                hidden_dim=cfg.modules.projector_hidden, message_dim=msg_dim,
                output_dim=cfg.modules.adapter_hidden, adapter_hidden=cfg.modules.adapter_hidden),
            action_head=ContinuousActionHead(
                cfg.modules.adapter_hidden, action_dim=5, hidden_dim=cfg.modules.action_hidden),
            obs_to_text_fn=listener_obs_to_text,
        ).to(device)
    else:
        lst_backbone = FrozenLLM(model_id, device, "float16")
        listener = ListenerAgent(
            backbone=lst_backbone, obs_projector=ObsProjector(
                lst_backbone.hidden_size, cfg.modules.projector_hidden, cfg.modules.projector_hidden),
            receiver_adapter=ReceiverAdapter(
                hidden_dim=cfg.modules.projector_hidden, message_dim=msg_dim,
                output_dim=cfg.modules.adapter_hidden, adapter_hidden=cfg.modules.adapter_hidden),
            action_head=ContinuousActionHead(
                cfg.modules.adapter_hidden, action_dim=5, hidden_dim=cfg.modules.action_hidden),
            obs_to_text_fn=listener_obs_to_text,
        ).to(device)

    # Use LoRA trainer if any agent has LoRA
    if lora_speaker and lora_listener:
        trainer = LoRAPPOTrainer(speaker, listener, cfg.training, device=device)
    elif lora_speaker:
        # Mixed: LoRA speaker + frozen listener
        # Use standard PPO but include speaker's LoRA params
        trainer = PPOTrainer(speaker, listener, cfg.training, device=device)
    elif lora_listener:
        trainer = PPOTrainer(speaker, listener, cfg.training, device=device)
    else:
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
    return float(np.mean(rewards)), float(np.std(rewards))


def run_experiment(
    label: str, cfg: Config, model_id: str, device: str,
    total_episodes: int, eval_every: int, eval_episodes: int,
    seed: int, lora_rank: int,
    lora_speaker: bool = False, lora_listener: bool = False,
    frozen_backbone=None,
):
    """Run one experiment configuration."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  comm={cfg.comm.type}, lora_spk={lora_speaker}, lora_lst={lora_listener}")
    print(f"  model={model_id}, seed={seed}, episodes={total_episodes}")
    print(f"{'='*60}")

    cfg.seed = seed
    set_all_seeds(seed)

    if not lora_speaker and not lora_listener:
        if frozen_backbone is None:
            frozen_backbone = FrozenLLM(model_id, device, "float16")
        speaker, listener, trainer = build_frozen_system(cfg, frozen_backbone, device)
    else:
        speaker, listener, trainer = build_lora_system(
            cfg, model_id, device, lora_rank, lora_speaker, lora_listener
        )

    env = SpeakerListenerEnv(max_cycles=cfg.env.max_cycles, continuous_actions=True)
    rollout_eps = cfg.training.rollout_episodes

    curve = {"episodes": [], "train_reward": [], "eval_reward": [], "eval_std": []}
    episode_count = 0
    t_start = time.time()

    while episode_count < total_episodes:
        buffer, rm = trainer.collect_rollouts(env, rollout_eps)
        um = trainer.update(buffer)
        episode_count += rollout_eps

        curve["episodes"].append(float(episode_count))
        curve["train_reward"].append(float(rm["episode_reward_mean"]))

        if episode_count % eval_every < rollout_eps:
            eval_mean, eval_std = evaluate(speaker, listener, env, eval_episodes)
            curve["eval_reward"].append(eval_mean)
            curve["eval_std"].append(eval_std)
            elapsed = time.time() - t_start
            remaining = elapsed / episode_count * (total_episodes - episode_count)
            lora_grad_info = ""
            if "speaker_lora_grad_norm" in um:
                lora_grad_info = f"  spk_lora_grad={um['speaker_lora_grad_norm']:.4f}"
            if "listener_lora_grad_norm" in um:
                lora_grad_info += f"  lst_lora_grad={um['listener_lora_grad_norm']:.4f}"
            print(
                f"  [{episode_count:>6}/{total_episodes}] "
                f"train={rm['episode_reward_mean']:>8.2f}  "
                f"eval={eval_mean:>8.2f} +/- {eval_std:.2f}"
                f"{lora_grad_info}  "
                f"elapsed={elapsed/60:.1f}m  ETA={remaining/60:.1f}m"
            )

            # Anneal Gumbel temperature
            if cfg.comm.type == "discrete":
                progress = episode_count / total_episodes
                new_tau = max(0.1, 1.0 * (1 - progress) + 0.1 * progress)
                speaker.comm.tau = new_tau
        else:
            curve["eval_reward"].append(None)
            curve["eval_std"].append(None)

    final_mean, final_std = evaluate(speaker, listener, env, eval_episodes)
    elapsed = time.time() - t_start
    print(f"  FINAL: eval={final_mean:.2f} +/- {final_std:.2f}, time={elapsed/60:.1f}min")

    env.close()

    # Clean up GPU memory
    del speaker, listener, trainer
    torch.cuda.empty_cache()

    return {
        "label": label,
        "comm_type": cfg.comm.type,
        "lora_speaker": lora_speaker,
        "lora_listener": lora_listener,
        "seed": seed,
        "final_eval_reward": final_mean,
        "final_eval_std": final_std,
        "time_minutes": elapsed / 60,
        "curve": curve,
    }


def main():
    parser = argparse.ArgumentParser(description="LoRA vs Frozen comparison")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--eval-episodes", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-135M-Instruct",
                        help="LLM model to use")
    parser.add_argument("--comm-methods", nargs="*", default=["vq_ssr", "ssr_v2", "discrete"],
                        help="Comm methods to test")
    parser.add_argument("--output", type=str, default="results/lora_comparison.json")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Define experiment grid
    # For each comm method: frozen, lora_speaker, lora_both
    backbone_configs = [
        ("frozen", False, False),
        ("lora_spk", True, False),
        ("lora_both", True, True),
    ]

    comm_configs = {}
    for method in args.comm_methods:
        if method == "vq_ssr":
            comm_configs[method] = CommConfig(type="vq_ssr", dim=8, num_codes=16)
        elif method == "ssr_v2":
            comm_configs[method] = CommConfig(type="ssr_v2", dim=8, normalize=False, residual=True)
        elif method == "ssr":
            comm_configs[method] = CommConfig(type="ssr", dim=8, normalize=True)
        elif method == "discrete":
            comm_configs[method] = CommConfig(type="discrete", dim=8, num_symbols=8)
        elif method == "continuous":
            comm_configs[method] = CommConfig(type="continuous", dim=8)
        elif method == "no_comm":
            comm_configs[method] = CommConfig(type="none", dim=1)

    all_results = []
    total_start = time.time()

    # Load frozen backbone once for all frozen experiments
    print(f"Model: {args.model}")
    frozen_backbone = FrozenLLM(args.model, device, "float16")

    for comm_name, comm_cfg in comm_configs.items():
        for bb_name, lora_spk, lora_lst in backbone_configs:
            # Skip LoRA-speaker with discrete (no gradient flow anyway — but keep it as control!)
            for seed in args.seeds:
                cfg = load_config(args.config)
                cfg.comm = comm_cfg
                label = f"{comm_name}_{bb_name}_s{seed}"

                result = run_experiment(
                    label=label, cfg=cfg, model_id=args.model, device=device,
                    total_episodes=args.episodes, eval_every=args.eval_every,
                    eval_episodes=args.eval_episodes, seed=seed,
                    lora_rank=args.lora_rank,
                    lora_speaker=lora_spk, lora_listener=lora_lst,
                    frozen_backbone=frozen_backbone if not lora_spk and not lora_lst else None,
                )
                all_results.append(result)

    total_time = (time.time() - total_start) / 60
    print(f"\nTotal time: {total_time:.1f} minutes")

    # Summary
    print(f"\n{'='*70}")
    print(f"  LORA COMPARISON SUMMARY")
    print(f"{'='*70}")
    print(f"{'Label':<35} {'Eval Reward':>14} {'Time (min)':>10}")
    print(f"{'-'*35} {'-'*14} {'-'*10}")

    sorted_results = sorted(all_results, key=lambda r: r["final_eval_reward"], reverse=True)
    for r in sorted_results:
        reward_str = f"{r['final_eval_reward']:.2f} +/- {r['final_eval_std']:.2f}"
        print(f"{r['label']:<35} {reward_str:>14} {r['time_minutes']:>8.1f}")

    # The key finding: does LoRA-speaker + differentiable comm > LoRA-speaker + discrete?
    print(f"\n  KEY COMPARISON: Does differentiable comm enable better LoRA adaptation?")
    for comm_name in comm_configs:
        frozen_r = [r for r in all_results if r["comm_type"] == comm_configs[comm_name].type and not r["lora_speaker"]]
        lora_r = [r for r in all_results if r["comm_type"] == comm_configs[comm_name].type and r["lora_speaker"]]
        if frozen_r and lora_r:
            frozen_mean = np.mean([r["final_eval_reward"] for r in frozen_r])
            lora_mean = np.mean([r["final_eval_reward"] for r in lora_r])
            delta = lora_mean - frozen_mean
            print(f"  {comm_name}: frozen={frozen_mean:.2f}, lora={lora_mean:.2f}, delta={delta:+.2f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        "config": {
            "model": args.model,
            "lora_rank": args.lora_rank,
            "episodes": args.episodes,
            "seeds": args.seeds,
        },
        "results": all_results,
        "total_time_minutes": total_time,
    }

    for r in save_data["results"]:
        r["curve"] = {
            k: [float(x) if x is not None else None for x in v]
            for k, v in r["curve"].items()
        }

    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
