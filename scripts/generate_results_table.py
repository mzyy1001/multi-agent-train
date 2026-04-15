#!/usr/bin/env python3
"""Generate LaTeX results table from comparison_v2.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main():
    results_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("results/comparison_v2.json")

    if not results_path.exists():
        print(f"Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as f:
        data = json.load(f)

    aggregated = data["aggregated"]

    # Sort by mean reward (descending)
    aggregated.sort(key=lambda x: x["mean_reward"], reverse=True)

    # Find no_comm baseline
    no_comm = next((a for a in aggregated if a["method"] == "no_comm"), None)
    no_comm_reward = no_comm["mean_reward"] if no_comm else None

    # Find discrete baseline
    discrete = next((a for a in aggregated if a["method"] == "discrete"), None)
    discrete_reward = discrete["mean_reward"] if discrete else None

    print("% Auto-generated results table")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Final evaluation reward (mean $\\pm$ std across seeds). Higher = better.}")
    print("\\vspace{0.3em}")
    print("\\begin{tabular}{@{}lccc@{}}")
    print("\\toprule")
    print("\\textbf{Method} & \\textbf{Reward} & \\textbf{vs.\\ No-Comm} & \\textbf{vs.\\ Discrete} \\\\")
    print("\\midrule")

    method_labels = {
        "vq_ssr": "VQ-SSR ($d{=}8$, $C{=}16$)",
        "ssr_v2": "SSR v2 ($d{=}8$)",
        "discrete": "Discrete (Gumbel, $K{=}8$)",
        "ssr": "SSR ($d{=}8$, with LN)",
        "ssr_no_ln": "SSR No-LN ($d{=}8$)",
        "continuous": "Continuous ($d{=}8$)",
        "no_comm": "No Communication",
    }

    best_method = aggregated[0]["method"]

    for agg in aggregated:
        method = agg["method"]
        label = method_labels.get(method, method)
        reward = agg["mean_reward"]
        std = agg["std_reward"]

        # Bold best method
        if method == best_method:
            reward_str = f"$\\mathbf{{{reward:.2f} \\pm {std:.2f}}}$"
        else:
            reward_str = f"${reward:.2f} \\pm {std:.2f}$"

        vs_nocomm = f"$+{reward - no_comm_reward:.2f}$" if no_comm_reward is not None and method != "no_comm" else "---"

        if method == "discrete":
            vs_discrete = "---"
        elif discrete_reward is not None:
            diff = reward - discrete_reward
            vs_discrete = f"${diff:+.2f}$"
        else:
            vs_discrete = "---"

        print(f"{label} & {reward_str} & {vs_nocomm} & {vs_discrete} \\\\")

    print("\\bottomrule")
    print("\\end{tabular}")
    print("\\end{table}")

    # Print summary stats
    print("\n% Summary")
    print(f"% Best method: {best_method} ({aggregated[0]['mean_reward']:.2f})")
    if no_comm:
        best_improvement = aggregated[0]["mean_reward"] - no_comm_reward
        print(f"% Improvement over no-comm: {best_improvement:+.2f}")

    # Message analysis summary
    print("\n% Message Analysis")
    for agg in aggregated:
        method = agg["method"]
        if method == "no_comm":
            continue
        # Check if message stats exist in per-seed results
        for seed_result in agg.get("per_seed", []):
            msg_stats = seed_result.get("message_stats", {})
            if msg_stats:
                print(f"% {method} (seed {seed_result['seed']}): "
                      f"msg_var={msg_stats.get('overall_var', 'N/A'):.4f}, "
                      f"msg_norm={msg_stats.get('norm_mean', 'N/A'):.4f}")
                break


if __name__ == "__main__":
    main()
