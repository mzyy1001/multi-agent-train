#!/usr/bin/env python3
"""Launch a grid of experiments across comm types and SSR dimensions."""

from __future__ import annotations

import argparse
import subprocess
import sys
from itertools import product


COMM_CONFIGS = {
    "ssr": "comm/ssr.yaml",
    "discrete": "comm/discrete.yaml",
    "continuous": "comm/continuous.yaml",
    "none": "comm/none.yaml",
}

MODEL_CONFIGS = {
    "smol_smol": "model/smol_smol.yaml",
    "smol_qwen": "model/smol_qwen.yaml",
}

SSR_DIMS = [4, 8, 16]
SEEDS = [42, 123, 456]


def main():
    parser = argparse.ArgumentParser(description="Sweep experiments")
    parser.add_argument(
        "--comms", nargs="*", default=list(COMM_CONFIGS.keys()),
        help="Communication types to sweep",
    )
    parser.add_argument(
        "--models", nargs="*", default=["smol_smol"],
        help="Model pairs to sweep",
    )
    parser.add_argument(
        "--dims", nargs="*", type=int, default=SSR_DIMS,
        help="SSR dimensions to sweep (only for ssr/continuous)",
    )
    parser.add_argument(
        "--seeds", nargs="*", type=int, default=SEEDS,
        help="Seeds to sweep",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands only")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    commands = []

    for comm, model, seed in product(args.comms, args.models, args.seeds):
        if comm in ("ssr", "continuous"):
            dims = args.dims
        else:
            dims = [8]  # dim doesn't matter for discrete/none

        for dim in dims:
            run_name = f"{comm}_d{dim}_{model}_s{seed}"
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", "configs/default.yaml",
                "--overrides", COMM_CONFIGS[comm], MODEL_CONFIGS[model],
                "--seed", str(seed),
                "--device", args.device,
                "--run-name", run_name,
                "--set", f"comm.dim={dim}",
            ]
            commands.append((run_name, cmd))

    print(f"Total experiments: {len(commands)}")
    for name, cmd in commands:
        print(f"  {name}: {' '.join(cmd)}")

    if args.dry_run:
        return

    for name, cmd in commands:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print(f"{'='*60}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"FAILED: {name} (exit code {result.returncode})")


if __name__ == "__main__":
    main()
