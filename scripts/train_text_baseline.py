#!/usr/bin/env python3
"""Baseline: Standard NLI where classifier sees BOTH premise and hypothesis.

No communication channel at all — just a single LLM that reads
"Premise: ... Hypothesis: ..." and classifies.

This is the upper bound for the cooperative task: if SSR/VQ-SSR can't
approach this, the communication bottleneck is too restrictive.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbone.llm import FrozenLLM
from src.backbone.lora_llm import LoRALLM
from src.env_text_coop_snli import SNLICoopDataset


class TextBaseline(nn.Module):
    """Single-agent NLI: LLM reads premise+hypothesis, classifies."""

    def __init__(self, backbone, hidden_dim=128, is_lora=False):
        super().__init__()
        self.is_lora = is_lora
        if is_lora:
            self.backbone = backbone
        else:
            self._backbone = backbone

        h_size = backbone.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, premises: list[str], hypotheses: list[str], training: bool = False):
        # Concatenate premise + hypothesis into one input
        texts = [
            f"Premise: {p} Hypothesis: {h}"
            for p, h in zip(premises, hypotheses)
        ]
        backbone = self.backbone if self.is_lora else self._backbone
        if self.is_lora and training:
            h = backbone.encode_batch(texts).float()
        else:
            with torch.no_grad():
                h = backbone.encode_batch(texts).float()
        logits = self.classifier(h).squeeze(-1)
        return logits


def train_epoch(model, dataset, optimizer, device, batch_size=32):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = max(1, len(dataset) // batch_size)

    for _ in range(n_batches):
        batch = dataset.sample_batch(batch_size)
        premises = [s.context for s in batch]
        hypotheses = [s.question for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)

        logits = model(premises, hypotheses, training=True)
        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += len(batch)

    return {"train_loss": total_loss / n_batches, "train_acc": correct / total}


def evaluate(model, dataset, device, batch_size=64):
    model.eval()
    dataset.reset()
    correct = 0
    total = 0

    with torch.no_grad():
        n_batches = max(1, len(dataset) // batch_size)
        for _ in range(n_batches):
            batch = dataset.get_batch(batch_size)
            premises = [s.context for s in batch]
            hypotheses = [s.question for s in batch]
            labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)

            logits = model(premises, hypotheses, training=False)
            preds = (logits > 0).float()
            correct += (preds == labels).sum().item()
            total += len(batch)

    return {"val_acc": correct / total}


def run_experiment(label, model_id, device, is_lora, lora_rank, epochs, lr, seed):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  model={model_id}, lora={is_lora}, seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = SNLICoopDataset("train", max_samples=10000, seed=seed)
    val_data = SNLICoopDataset("val", max_samples=2000, seed=seed)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    if is_lora:
        backbone = LoRALLM(model_id, device, "float16", lora_rank=lora_rank)
    else:
        backbone = FrozenLLM(model_id, device, "float16")

    model = TextBaseline(backbone, hidden_dim=128, is_lora=is_lora).to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr
    )

    t_start = time.time()
    history = {"train_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_m = train_epoch(model, train_data, optimizer, device)
        val_m = evaluate(model, val_data, device)
        history["train_loss"].append(train_m["train_loss"])
        history["train_acc"].append(train_m["train_acc"])
        history["val_acc"].append(val_m["val_acc"])

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:>3}: loss={train_m['train_loss']:.4f} "
                  f"train_acc={train_m['train_acc']:.4f} "
                  f"val_acc={val_m['val_acc']:.4f} elapsed={elapsed:.1f}s")

    final = evaluate(model, val_data, device)
    elapsed = time.time() - t_start
    print(f"  FINAL: val_acc={final['val_acc']:.4f}, time={elapsed:.1f}s")

    del model, optimizer
    torch.cuda.empty_cache()

    return {
        "label": label,
        "lora": is_lora,
        "seed": seed,
        "final_val_acc": final["val_acc"],
        "time_seconds": elapsed,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    parser.add_argument("--output", default="results/text_baseline_snli.json")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    all_results = []

    for seed in args.seeds:
        # Frozen baseline
        r = run_experiment(
            f"text_transfer_frozen_s{seed}", args.model, device,
            is_lora=False, lora_rank=args.lora_rank,
            epochs=args.epochs, lr=args.lr, seed=seed,
        )
        all_results.append(r)

        # LoRA baseline
        r = run_experiment(
            f"text_transfer_lora_s{seed}", args.model, device,
            is_lora=True, lora_rank=args.lora_rank,
            epochs=args.epochs, lr=args.lr, seed=seed,
        )
        all_results.append(r)

    # Summary
    frozen_accs = [r["final_val_acc"] for r in all_results if not r["lora"]]
    lora_accs = [r["final_val_acc"] for r in all_results if r["lora"]]
    print(f"\n{'='*60}")
    print(f"  TEXT TRANSFER BASELINE (upper bound)")
    print(f"  Frozen: {np.mean(frozen_accs):.4f} +/- {np.std(frozen_accs):.4f}")
    print(f"  LoRA:   {np.mean(lora_accs):.4f} +/- {np.std(lora_accs):.4f}")
    print(f"{'='*60}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": all_results}, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
