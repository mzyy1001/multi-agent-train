#!/usr/bin/env python3
"""Train cooperative text-grounded classification with differentiable communication.

Architecture:
  Agent A (Encoder): context → LLM → projector → comm_channel → message
  Agent B (Classifier): question → LLM → projector → adapter(z, message) → classifier → P(label)

Loss: Binary cross-entropy on Agent B's classification output.
Gradient flow: BCE loss → classifier → adapter → message → comm_channel → projector_A → LLM_A (via LoRA)

This directly tests the gradient highway thesis:
- With frozen LLM: only projector/comm/adapter learn from the loss
- With LoRA on Agent A: the encoder LLM adapts its representations to better serve Agent B
- With discrete comm: gradients reach Agent A's LLM via STE but with bias
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.backbone.llm import FrozenLLM
from src.backbone.lora_llm import LoRALLM
from src.comm import build_comm_channel
from src.comm.base import CommChannel
from src.config import CommConfig
from src.env_text_coop_snli import SNLICoopDataset as TextCoopDataset, TextCoopSample
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter


class TextEncoder(nn.Module):
    """Agent A: encodes context text through LLM + projector + comm channel."""

    def __init__(self, backbone, projector, comm_channel, is_lora=False):
        super().__init__()
        self.projector = projector
        self.comm = comm_channel
        self.is_lora = is_lora
        if is_lora:
            self.backbone = backbone  # Register as submodule
        else:
            self._backbone = backbone  # Keep outside parameters()

    def forward(self, texts: list[str], training: bool = False) -> torch.Tensor:
        backbone = self.backbone if self.is_lora else self._backbone
        if self.is_lora and training:
            h = backbone.encode_batch(texts).float()
        else:
            with torch.no_grad():
                h = backbone.encode_batch(texts).float()
        z = self.projector(h)
        message = self.comm(z)
        return message


class TextClassifier(nn.Module):
    """Agent B: receives question + message, produces binary classification."""

    def __init__(self, backbone, projector, adapter, hidden_dim, is_lora=False):
        super().__init__()
        self.projector = projector
        self.adapter = adapter
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.is_lora = is_lora
        if is_lora:
            self.backbone = backbone
        else:
            self._backbone = backbone

    def forward(self, texts: list[str], message: torch.Tensor, training: bool = False) -> torch.Tensor:
        backbone = self.backbone if self.is_lora else self._backbone
        if self.is_lora and training:
            h = backbone.encode_batch(texts).float()
        else:
            with torch.no_grad():
                h = backbone.encode_batch(texts).float()
        z = self.projector(h)
        h_tilde = self.adapter(z, message)
        logits = self.classifier(h_tilde).squeeze(-1)
        return logits


def build_system(
    model_id: str, device: str, comm_cfg: CommConfig,
    lora_encoder: bool, lora_classifier: bool, lora_rank: int,
    proj_dim: int = 128, adapter_dim: int = 128,
):
    """Build encoder + classifier system."""
    # Build encoder backbone
    if lora_encoder:
        enc_backbone = LoRALLM(model_id, device, "float16", lora_rank=lora_rank)
    else:
        enc_backbone = FrozenLLM(model_id, device, "float16")

    # Build classifier backbone
    if lora_classifier:
        cls_backbone = LoRALLM(model_id, device, "float16", lora_rank=lora_rank)
    else:
        cls_backbone = FrozenLLM(model_id, device, "float16")

    hidden_size = enc_backbone.hidden_size

    # Build comm channel
    comm_channel = build_comm_channel(comm_cfg, input_dim=proj_dim)
    msg_dim = comm_channel.message_dim()

    encoder = TextEncoder(
        backbone=enc_backbone,
        projector=ObsProjector(hidden_size, proj_dim, proj_dim),
        comm_channel=comm_channel,
        is_lora=lora_encoder,
    ).to(device)

    classifier = TextClassifier(
        backbone=cls_backbone,
        projector=ObsProjector(hidden_size, proj_dim, proj_dim),
        adapter=ReceiverAdapter(
            hidden_dim=proj_dim, message_dim=msg_dim,
            output_dim=adapter_dim, adapter_hidden=adapter_dim,
        ),
        hidden_dim=adapter_dim,
        is_lora=lora_classifier,
    ).to(device)

    return encoder, classifier


def train_epoch(encoder, classifier, dataset, optimizer, device, batch_size=32):
    encoder.train()
    classifier.train()
    total_loss = 0.0
    correct = 0
    total = 0

    n_batches = max(1, len(dataset) // batch_size)
    for _ in range(n_batches):
        batch = dataset.sample_batch(batch_size)
        contexts = [s.context for s in batch]
        questions = [s.question for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)

        # Forward: encoder → message → classifier
        messages = encoder(contexts, training=True)
        logits = classifier(questions, messages, training=True)

        loss = F.binary_cross_entropy_with_logits(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(classifier.parameters()), 1.0
        )
        optimizer.step()

        total_loss += loss.item()
        preds = (logits > 0).float()
        correct += (preds == labels).sum().item()
        total += len(batch)

    return {
        "train_loss": total_loss / n_batches,
        "train_acc": correct / total,
    }


def evaluate(encoder, classifier, dataset, device, batch_size=64):
    encoder.eval()
    classifier.eval()
    dataset.reset()
    correct = 0
    total = 0

    with torch.no_grad():
        n_batches = max(1, len(dataset) // batch_size)
        for _ in range(n_batches):
            batch = dataset.get_batch(batch_size)
            contexts = [s.context for s in batch]
            questions = [s.question for s in batch]
            labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)

            messages = encoder(contexts, training=False)
            logits = classifier(questions, messages, training=False)

            preds = (logits > 0).float()
            correct += (preds == labels).sum().item()
            total += len(batch)

    return {"val_acc": correct / total}


def run_experiment(
    label: str, model_id: str, device: str, comm_cfg: CommConfig,
    lora_encoder: bool, lora_classifier: bool, lora_rank: int,
    epochs: int, lr: float, seed: int,
):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"  comm={comm_cfg.type}, lora_enc={lora_encoder}, lora_cls={lora_classifier}")
    print(f"  model={model_id}, epochs={epochs}, seed={seed}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_data = TextCoopDataset("train", max_samples=10000, seed=seed)
    val_data = TextCoopDataset("val", max_samples=2000, seed=seed)
    print(f"  Train: {len(train_data)} samples, Val: {len(val_data)} samples")

    encoder, classifier = build_system(
        model_id, device, comm_cfg,
        lora_encoder, lora_classifier, lora_rank,
    )

    all_params = list(encoder.parameters()) + list(classifier.parameters())
    trainable = sum(p.numel() for p in all_params if p.requires_grad)
    print(f"  Total trainable params: {trainable:,}")

    optimizer = torch.optim.Adam(
        [p for p in all_params if p.requires_grad], lr=lr
    )

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    t_start = time.time()

    for epoch in range(1, epochs + 1):
        train_metrics = train_epoch(encoder, classifier, train_data, optimizer, device)
        val_metrics = evaluate(encoder, classifier, val_data, device)

        history["train_loss"].append(train_metrics["train_loss"])
        history["train_acc"].append(train_metrics["train_acc"])
        history["val_acc"].append(val_metrics["val_acc"])

        if epoch % 5 == 0 or epoch == 1:
            elapsed = time.time() - t_start
            print(f"  Epoch {epoch:>3}: loss={train_metrics['train_loss']:.4f} "
                  f"train_acc={train_metrics['train_acc']:.4f} "
                  f"val_acc={val_metrics['val_acc']:.4f} "
                  f"elapsed={elapsed:.1f}s")

    final_val = evaluate(encoder, classifier, val_data, device)
    elapsed = time.time() - t_start
    print(f"  FINAL: val_acc={final_val['val_acc']:.4f}, time={elapsed:.1f}s")

    del encoder, classifier, optimizer
    torch.cuda.empty_cache()

    return {
        "label": label,
        "comm_type": comm_cfg.type,
        "lora_encoder": lora_encoder,
        "lora_classifier": lora_classifier,
        "seed": seed,
        "final_val_acc": final_val["val_acc"],
        "time_seconds": elapsed,
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser(description="Text cooperative task: gradient highway test")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--seeds", nargs="*", type=int, default=[42])
    parser.add_argument("--output", type=str, default="results/text_coop_snli.json")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Experiment grid
    configs = [
        # (label_prefix, comm_cfg, lora_enc, lora_cls)
        ("ssr_frozen", CommConfig(type="ssr", dim=8, normalize=False), False, False),
        ("ssr_lora_enc", CommConfig(type="ssr", dim=8, normalize=False), True, False),
        ("ssr_lora_both", CommConfig(type="ssr", dim=8, normalize=False), True, True),
        ("vqssr_frozen", CommConfig(type="vq_ssr", dim=8, num_codes=16), False, False),
        ("vqssr_lora_enc", CommConfig(type="vq_ssr", dim=8, num_codes=16), True, False),
        ("vqssr_lora_both", CommConfig(type="vq_ssr", dim=8, num_codes=16), True, True),
        ("discrete_frozen", CommConfig(type="discrete", dim=8, num_symbols=8), False, False),
        ("discrete_lora_enc", CommConfig(type="discrete", dim=8, num_symbols=8), True, False),
        ("discrete_lora_both", CommConfig(type="discrete", dim=8, num_symbols=8), True, True),
        ("nocomm_frozen", CommConfig(type="none", dim=1), False, False),
    ]

    all_results = []
    total_start = time.time()

    for label_prefix, comm_cfg, lora_enc, lora_cls in configs:
        for seed in args.seeds:
            label = f"{label_prefix}_s{seed}"
            result = run_experiment(
                label=label, model_id=args.model, device=device,
                comm_cfg=comm_cfg, lora_encoder=lora_enc, lora_classifier=lora_cls,
                lora_rank=args.lora_rank, epochs=args.epochs, lr=args.lr, seed=seed,
            )
            all_results.append(result)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time/60:.1f} minutes")

    # Summary
    print(f"\n{'='*70}")
    print(f"  TEXT COOPERATIVE TASK — GRADIENT HIGHWAY RESULTS")
    print(f"{'='*70}")
    print(f"{'Label':<30} {'Val Acc':>10} {'Time (s)':>10}")
    print(f"{'-'*30} {'-'*10} {'-'*10}")

    sorted_results = sorted(all_results, key=lambda r: r["final_val_acc"], reverse=True)
    for r in sorted_results:
        print(f"{r['label']:<30} {r['final_val_acc']:>8.4f} {r['time_seconds']:>8.1f}")

    # Key comparison
    print(f"\n  KEY: Does differentiable comm + LoRA-encoder outperform discrete + LoRA-encoder?")
    for comm in ["ssr", "vq_ssr", "discrete"]:
        frozen = [r for r in all_results if r["comm_type"] == comm and not r["lora_encoder"] and not r["lora_classifier"]]
        lora_enc = [r for r in all_results if r["comm_type"] == comm and r["lora_encoder"] and not r["lora_classifier"]]
        if frozen and lora_enc:
            f_acc = np.mean([r["final_val_acc"] for r in frozen])
            l_acc = np.mean([r["final_val_acc"] for r in lora_enc])
            delta = l_acc - f_acc
            print(f"  {comm}: frozen={f_acc:.4f}, lora_enc={l_acc:.4f}, delta={delta:+.4f}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"config": vars(args), "results": all_results, "total_time": total_time}, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
