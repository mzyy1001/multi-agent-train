#!/usr/bin/env python3
"""Generate publication-quality figures from comparison_v2.json results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHOD_COLORS = {
    "vq_ssr": "#e63946",
    "ssr_v2": "#457b9d",
    "discrete": "#2a9d8f",
    "ssr": "#264653",
    "ssr_no_ln": "#f4a261",
    "continuous": "#a8dadc",
    "no_comm": "#999999",
}

METHOD_LABELS = {
    "vq_ssr": "VQ-SSR (ours)",
    "ssr_v2": "SSR v2 (ours)",
    "discrete": "Discrete (Gumbel)",
    "ssr": "SSR (original)",
    "ssr_no_ln": "SSR No-LN",
    "continuous": "Continuous",
    "no_comm": "No Communication",
}

METHOD_ORDER = ["vq_ssr", "ssr_v2", "discrete", "ssr_no_ln", "ssr", "continuous", "no_comm"]


def smooth(values, window=20):
    """Simple moving average smoothing, handling None values."""
    clean = [v for v in values if v is not None]
    if len(clean) < window:
        return clean
    kernel = np.ones(window) / window
    return np.convolve(clean, kernel, mode="valid").tolist()


def plot_learning_curves(data, output_dir):
    """Plot eval reward curves for all methods, averaged across seeds."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for method_name in METHOD_ORDER:
        agg = next((a for a in data["aggregated"] if a["method"] == method_name), None)
        if agg is None:
            continue

        # Collect eval curves across seeds
        all_eval_rewards = []
        episodes = None
        for seed_result in agg["per_seed"]:
            curve = seed_result["curve"]
            eval_rewards = [r for r in curve["eval_reward"] if r is not None]
            eval_eps = [e for e, r in zip(curve["episodes"], curve["eval_reward"]) if r is not None]
            if eval_rewards:
                all_eval_rewards.append(eval_rewards)
                if episodes is None:
                    episodes = eval_eps

        if not all_eval_rewards or episodes is None:
            continue

        # Align lengths
        min_len = min(len(r) for r in all_eval_rewards)
        all_eval_rewards = [r[:min_len] for r in all_eval_rewards]
        episodes = episodes[:min_len]

        mean_rewards = np.mean(all_eval_rewards, axis=0)
        std_rewards = np.std(all_eval_rewards, axis=0)

        color = METHOD_COLORS.get(method_name, "#333333")
        label = METHOD_LABELS.get(method_name, method_name)

        ax.plot(episodes, mean_rewards, color=color, label=label, linewidth=2)
        ax.fill_between(episodes, mean_rewards - std_rewards, mean_rewards + std_rewards,
                        alpha=0.15, color=color)

    ax.set_xlabel("Training Episodes", fontsize=12)
    ax.set_ylabel("Eval Reward (higher = better)", fontsize=12)
    ax.set_title("Learning Curves: Communication Methods Comparison", fontsize=14)
    ax.legend(loc="lower right", fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.2)

    plt.tight_layout()
    fig.savefig(output_dir / "learning_curves.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "learning_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved learning_curves.pdf/png")


def plot_bar_chart(data, output_dir):
    """Bar chart of final rewards with error bars."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))

    methods = []
    means = []
    stds = []
    colors = []

    # Sort by mean reward
    sorted_agg = sorted(data["aggregated"], key=lambda x: x["mean_reward"], reverse=True)

    for agg in sorted_agg:
        method = agg["method"]
        methods.append(METHOD_LABELS.get(method, method))
        means.append(agg["mean_reward"])
        stds.append(agg["std_reward"])
        colors.append(METHOD_COLORS.get(method, "#333333"))

    x = np.arange(len(methods))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right", fontsize=10)
    ax.set_ylabel("Eval Reward (mean over 3 seeds)", fontsize=12)
    ax.set_title("Final Performance Comparison", fontsize=14)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels on bars
    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + std + 2,
                f"{mean:.1f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_dir / "bar_comparison.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "bar_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved bar_comparison.pdf/png")


def plot_message_analysis(data, output_dir):
    """Plot message statistics for each method."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    methods_with_stats = []
    for agg in data["aggregated"]:
        if agg["method"] == "no_comm":
            continue
        seed_result = agg["per_seed"][0]  # Use first seed
        msg_stats = seed_result.get("message_stats", {})
        if msg_stats and "per_dim_var" in msg_stats:
            methods_with_stats.append((agg["method"], msg_stats))

    if not methods_with_stats:
        plt.close(fig)
        return

    # Per-dimension variance heatmap
    ax = axes[0]
    method_names = []
    variance_matrix = []
    for method, stats in methods_with_stats:
        method_names.append(METHOD_LABELS.get(method, method))
        variance_matrix.append(stats["per_dim_var"])

    variance_matrix = np.array(variance_matrix)
    im = ax.imshow(variance_matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, fontsize=9)
    ax.set_xlabel("Message Dimension", fontsize=11)
    ax.set_title("Per-Dimension Variance", fontsize=12)
    fig.colorbar(im, ax=ax)

    # Message norm distribution
    ax = axes[1]
    for method, stats in methods_with_stats:
        label = METHOD_LABELS.get(method, method)
        color = METHOD_COLORS.get(method, "#333333")
        norm_mean = stats.get("norm_mean", 0)
        norm_std = stats.get("norm_std", 0)
        ax.barh(label, norm_mean, xerr=norm_std, color=color, edgecolor="black",
                linewidth=0.5, capsize=3)
    ax.set_xlabel("Message Norm", fontsize=11)
    ax.set_title("Message Magnitude", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / "message_analysis.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / "message_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved message_analysis.pdf/png")


def main():
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/comparison_v2.json")
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("figures/")
    output_dir.mkdir(parents=True, exist_ok=True)

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    print("Generating figures...")
    plot_learning_curves(data, output_dir)
    plot_bar_chart(data, output_dir)
    plot_message_analysis(data, output_dir)
    print("Done!")


if __name__ == "__main__":
    main()
