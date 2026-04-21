#!/usr/bin/env python3
"""Multi-Agent Company RL Training with Dual-Channel Communication.

Trains CEO + CTO agents on the market simulation using PPO.
Compares: no-comm, text-only, SSR-only, dual-channel.

Usage:
    PYTHONPATH=. python scripts/train_company.py --comm dual --episodes 5000 --seed 42
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Handle offline mode for servers without HF access
os.environ.setdefault("TRANSFORMERS_OFFLINE", "0")

from src.env_market_sim import MarketSimEnv, ScenarioConfig


class ObsEncoder(nn.Module):
    """Encode numeric observation into a latent vector."""
    def __init__(self, obs_dim, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self, x):
        return self.net(x)


class SSRChannel(nn.Module):
    """SSR communication channel: encode latent → message vector."""
    def __init__(self, input_dim, msg_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, msg_dim * 4), nn.GELU(),
            nn.Linear(msg_dim * 4, msg_dim),
        )
        self.msg_dim = msg_dim
    def forward(self, z):
        return self.encoder(z)


class MessageAdapter(nn.Module):
    """Fuse own latent with received messages."""
    def __init__(self, own_dim, msg_dim, output_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(own_dim + msg_dim, output_dim), nn.LayerNorm(output_dim), nn.GELU(),
            nn.Linear(output_dim, output_dim),
        )
    def forward(self, z_own, message):
        return self.net(torch.cat([z_own, message], dim=-1))


class ContinuousActionHead(nn.Module):
    """Policy + value head for continuous actions."""
    LOG_STD_MIN, LOG_STD_MAX = -3.0, 1.0

    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU())
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)
        self.value = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, h):
        feat = self.shared(h)
        mean = self.mean(feat)
        log_std = self.log_std(feat).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std, self.value(h).squeeze(-1)

    def get_action(self, h, deterministic=False):
        mean, log_std, value = self.forward(h)
        std = log_std.exp()
        dist = Normal(mean, std)
        raw = mean if deterministic else dist.rsample()
        action = torch.sigmoid(raw)
        log_prob = dist.log_prob(raw).sum(-1) - torch.log(action * (1 - action) + 1e-6).sum(-1)
        return {"action": action, "raw": raw, "log_prob": log_prob, "value": value, "entropy": dist.entropy().sum(-1)}

    def evaluate(self, h, raw_action):
        mean, log_std, value = self.forward(h)
        std = log_std.exp()
        dist = Normal(mean, std)
        action = torch.sigmoid(raw_action)
        log_prob = dist.log_prob(raw_action).sum(-1) - torch.log(action * (1 - action) + 1e-6).sum(-1)
        return {"log_prob": log_prob, "value": value, "entropy": dist.entropy().sum(-1)}


class CompanyAgent(nn.Module):
    """Single agent in the company (CEO or CTO)."""
    def __init__(self, obs_dim, action_dim, msg_dim=32, hidden_dim=128):
        super().__init__()
        self.obs_encoder = ObsEncoder(obs_dim, hidden_dim)
        self.ssr_channel = SSRChannel(hidden_dim, msg_dim)
        self.adapter = MessageAdapter(hidden_dim, msg_dim, hidden_dim)
        self.action_head = ContinuousActionHead(hidden_dim, action_dim)
        self.msg_dim = msg_dim

    def encode(self, obs):
        return self.obs_encoder(obs)

    def send_message(self, z):
        return self.ssr_channel(z)

    def receive_and_act(self, z, message, deterministic=False):
        h = self.adapter(z, message)
        return self.action_head.get_action(h, deterministic)

    def evaluate_action(self, z, message, raw_action):
        h = self.adapter(z, message)
        return self.action_head.evaluate(h, raw_action)


def collect_episode(env, agents, comm_mode, device, deterministic=False):
    """Run one episode, return transitions."""
    obs = env.reset()
    transitions = []
    done = False

    while not done:
        ceo_obs_t = torch.tensor(obs["ceo"], dtype=torch.float32, device=device).unsqueeze(0)
        cto_obs_t = torch.tensor(obs["cto"], dtype=torch.float32, device=device).unsqueeze(0)

        with torch.no_grad():
            z_ceo = agents["ceo"].encode(ceo_obs_t)
            z_cto = agents["cto"].encode(cto_obs_t)

            # Communication
            if comm_mode in ("ssr", "dual"):
                msg_ceo = agents["ceo"].send_message(z_ceo)  # CEO → CTO
                msg_cto = agents["cto"].send_message(z_cto)  # CTO → CEO
            else:
                msg_ceo = torch.zeros(1, agents["ceo"].msg_dim, device=device)
                msg_cto = torch.zeros(1, agents["cto"].msg_dim, device=device)

            ceo_out = agents["ceo"].receive_and_act(z_ceo, msg_cto, deterministic)
            cto_out = agents["cto"].receive_and_act(z_cto, msg_ceo, deterministic)

        ceo_action = ceo_out["action"].squeeze(0).cpu().numpy()
        cto_action = cto_out["action"].squeeze(0).cpu().numpy()
        sales_action = np.array([0.0, 0.1, 0.5])  # Fixed sales for 2-agent setup

        obs_next, reward, done, info = env.step(ceo_action, cto_action, sales_action)

        transitions.append({
            "ceo_obs": obs["ceo"].copy(), "cto_obs": obs["cto"].copy(),
            "ceo_raw": ceo_out["raw"].squeeze(0).detach(),
            "cto_raw": cto_out["raw"].squeeze(0).detach(),
            "ceo_logp": ceo_out["log_prob"].item(),
            "cto_logp": cto_out["log_prob"].item(),
            "ceo_val": ceo_out["value"].item(),
            "cto_val": cto_out["value"].item(),
            "reward": reward, "done": done,
        })
        obs = obs_next

    return transitions, info


def ppo_update(agents, buffer, comm_mode, device, lr=3e-4, clip_eps=0.2,
               gamma=0.95, gae_lambda=0.95, epochs=4, batch_size=64):
    """PPO update for both agents jointly."""
    # Compute GAE
    rewards = [t["reward"] for t in buffer]
    values = [t["ceo_val"] for t in buffer]  # Use CEO value as shared baseline
    dones = [t["done"] for t in buffer]

    advantages, returns = [], []
    gae = 0.0
    for t in reversed(range(len(buffer))):
        nv = values[t + 1] if t + 1 < len(buffer) and not dones[t] else 0.0
        delta = rewards[t] + gamma * nv - values[t]
        gae = delta + gamma * gae_lambda * (1 - float(dones[t])) * gae
        advantages.insert(0, gae)
        returns.insert(0, gae + values[t])

    advantages = torch.tensor(advantages, dtype=torch.float32, device=device)
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

    ceo_obs = torch.tensor(np.array([t["ceo_obs"] for t in buffer]), dtype=torch.float32, device=device)
    cto_obs = torch.tensor(np.array([t["cto_obs"] for t in buffer]), dtype=torch.float32, device=device)
    ceo_raw = torch.stack([t["ceo_raw"] for t in buffer]).to(device)
    cto_raw = torch.stack([t["cto_raw"] for t in buffer]).to(device)
    old_ceo_logp = torch.tensor([t["ceo_logp"] for t in buffer], dtype=torch.float32, device=device)
    old_cto_logp = torch.tensor([t["cto_logp"] for t in buffer], dtype=torch.float32, device=device)

    all_params = list(agents["ceo"].parameters()) + list(agents["cto"].parameters())
    optimizer = torch.optim.Adam(all_params, lr=lr)
    n = len(buffer)
    metrics = defaultdict(float)
    updates = 0

    for _ in range(epochs):
        idx = np.random.permutation(n)
        for start in range(0, n, batch_size):
            mb = idx[start:start + batch_size]

            # Re-forward with gradients
            z_ceo = agents["ceo"].encode(ceo_obs[mb])
            z_cto = agents["cto"].encode(cto_obs[mb])

            if comm_mode in ("ssr", "dual"):
                msg_ceo = agents["ceo"].send_message(z_ceo)
                msg_cto = agents["cto"].send_message(z_cto)
            else:
                msg_ceo = torch.zeros(len(mb), agents["ceo"].msg_dim, device=device)
                msg_cto = torch.zeros(len(mb), agents["cto"].msg_dim, device=device)

            ceo_eval = agents["ceo"].evaluate_action(z_ceo, msg_cto, ceo_raw[mb])
            cto_eval = agents["cto"].evaluate_action(z_cto, msg_ceo, cto_raw[mb])

            # PPO loss for both agents
            for name, eval_out, old_lp in [("ceo", ceo_eval, old_ceo_logp[mb]),
                                             ("cto", cto_eval, old_cto_logp[mb])]:
                ratio = (eval_out["log_prob"] - old_lp).exp()
                s1 = ratio * advantages[mb]
                s2 = ratio.clamp(1 - clip_eps, 1 + clip_eps) * advantages[mb]
                policy_loss = -torch.min(s1, s2).mean()
                value_loss = F.mse_loss(eval_out["value"], returns_t[mb])
                entropy_loss = -eval_out["entropy"].mean()

                loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(all_params, 0.5)
                optimizer.step()

                metrics[f"{name}_policy_loss"] += policy_loss.item()
                metrics[f"{name}_value_loss"] += value_loss.item()

            # Message stats
            if comm_mode in ("ssr", "dual"):
                with torch.no_grad():
                    metrics["msg_var"] += msg_ceo.var().item()
                    metrics["msg_norm"] += msg_ceo.norm(dim=-1).mean().item()
            updates += 1

    return {k: v / max(updates, 1) for k, v in metrics.items()}


def train(comm_mode, n_episodes, scenario, seed, device, msg_dim=32, eval_every=200):
    """Train and evaluate."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MarketSimEnv(config=scenario, seed=seed)
    eval_env = MarketSimEnv(config=scenario, seed=seed + 10000)

    agents = {
        "ceo": CompanyAgent(env.CEO_OBS_DIM, env.CEO_ACT_DIM, msg_dim).to(device),
        "cto": CompanyAgent(env.CTO_OBS_DIM, env.CTO_ACT_DIM, msg_dim).to(device),
    }

    total_params = sum(sum(p.numel() for p in a.parameters()) for a in agents.values())
    print(f"  Total params: {total_params:,}", flush=True)

    curve = {"episodes": [], "train_profit": [], "eval_profit": []}
    buffer = []
    rollout_size = 16  # episodes per PPO update

    for ep in range(1, n_episodes + 1):
        transitions, info = collect_episode(env, agents, comm_mode, device)
        buffer.extend(transitions)

        if ep % rollout_size == 0 and len(buffer) > 0:
            metrics = ppo_update(agents, buffer, comm_mode, device)
            buffer = []

        if ep % eval_every == 0:
            # Evaluate
            eval_profits = []
            for eval_seed in range(5):
                eval_env.reset(seed=seed + 20000 + eval_seed)
                _, eval_info = collect_episode(eval_env, agents, comm_mode, device, deterministic=True)
                eval_profits.append(eval_info["cumulative_profit"])

            mean_profit = np.mean(eval_profits)
            std_profit = np.std(eval_profits)
            curve["episodes"].append(ep)
            curve["train_profit"].append(info["cumulative_profit"])
            curve["eval_profit"].append(mean_profit)

            print(f"  [{ep:>6}/{n_episodes}] train_profit={info['cumulative_profit']:.0f} "
                  f"eval_profit={mean_profit:.0f}±{std_profit:.0f} "
                  f"quality={info['product_quality']:.2f} debt={info['tech_debt']:.2f}",
                  flush=True)

    # Final eval across scenarios
    final_results = {}
    for sc_name, sc_cfg in [("medium", ScenarioConfig.medium()), ("hard", ScenarioConfig.hard()),
                              ("startup", ScenarioConfig.startup()), ("recession", ScenarioConfig.recession())]:
        profits = []
        for es in range(10):
            test_env = MarketSimEnv(config=sc_cfg, seed=seed + 30000 + es)
            test_env.reset()
            _, ti = collect_episode(test_env, agents, comm_mode, device, deterministic=True)
            profits.append(ti["cumulative_profit"])
        final_results[sc_name] = {"mean": float(np.mean(profits)), "std": float(np.std(profits))}
        print(f"  Scenario {sc_name}: profit={np.mean(profits):.0f}±{np.std(profits):.0f}", flush=True)

    return curve, final_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comm", default="dual", choices=["nocomm", "ssr", "dual"])
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--scenario", default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--msg-dim", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="results/company")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        args.device = "cpu"

    scenarios = {"easy": ScenarioConfig.easy(), "medium": ScenarioConfig.medium(),
                 "hard": ScenarioConfig.hard(), "startup": ScenarioConfig.startup(),
                 "recession": ScenarioConfig.recession()}
    scenario = scenarios.get(args.scenario, ScenarioConfig.medium())

    print(f"=== Company RL: comm={args.comm}, scenario={args.scenario}, seed={args.seed} ===", flush=True)

    curve, final_results = train(
        args.comm, args.episodes, scenario, args.seed, args.device, args.msg_dim
    )

    # Save
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "comm": args.comm, "scenario": args.scenario, "seed": args.seed,
        "final_results": final_results, "curve": curve,
    }
    out_path = out_dir / f"{args.comm}_{args.scenario}_s{args.seed}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {out_path}", flush=True)


if __name__ == "__main__":
    main()
