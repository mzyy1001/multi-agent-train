#!/usr/bin/env python3
"""Noisy Multi-Fact Retrieval: SSR vs Text Transfer under bandwidth + noise constraints.

Key comparisons:
1. SSR/VQ-SSR (d=32/64) through noisy channel — learned compression
2. Text transfer (full) — no noise, upper bound
3. Text transfer (truncated to N tokens) — bandwidth matched
4. Text transfer through noisy channel — text corrupted by noise

The hypothesis: SSR's learned compression + noise robustness should
outperform truncated text and noisy text, approaching full text transfer.
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
from src.comm import build_comm_channel
from src.comm.base import CommChannel
from src.config import CommConfig
from src.env_noisy_retrieval import NoisyRetrievalDataset, RetrievalSample
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter


class NoisyCommWrapper(nn.Module):
    """Wraps a comm channel and adds Gaussian noise to the output."""

    def __init__(self, channel: CommChannel, noise_std: float = 0.1):
        super().__init__()
        self.channel = channel
        self.noise_std = noise_std

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        m = self.channel(z)
        if self.training and self.noise_std > 0:
            m = m + torch.randn_like(m) * self.noise_std
        return m

    def message_dim(self) -> int:
        return self.channel.message_dim()


class SSREncoder(nn.Module):
    """Encoder: context -> LLM -> projector -> comm -> noisy message."""

    def __init__(self, backbone, projector, comm_channel, noise_std=0.1, is_lora=False):
        super().__init__()
        self.projector = projector
        self.comm = NoisyCommWrapper(comm_channel, noise_std)
        self.is_lora = is_lora
        if is_lora:
            self.backbone = backbone
        else:
            self._backbone = backbone

    def forward(self, texts, training=False):
        backbone = self.backbone if self.is_lora else self._backbone
        if self.is_lora and training:
            h = backbone.encode_batch(texts).float()
        else:
            with torch.no_grad():
                h = backbone.encode_batch(texts).float()
        z = self.projector(h)
        return self.comm(z)


class Classifier(nn.Module):
    """Classifier: question + message -> binary prediction."""

    def __init__(self, backbone, projector, adapter, hidden_dim, is_lora=False):
        super().__init__()
        self.projector = projector
        self.adapter = adapter
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.is_lora = is_lora
        if is_lora:
            self.backbone = backbone
        else:
            self._backbone = backbone

    def forward(self, texts, message, training=False):
        backbone = self.backbone if self.is_lora else self._backbone
        if self.is_lora and training:
            h = backbone.encode_batch(texts).float()
        else:
            with torch.no_grad():
                h = backbone.encode_batch(texts).float()
        z = self.projector(h)
        h_tilde = self.adapter(z, message)
        return self.head(h_tilde).squeeze(-1)


class TextTransferModel(nn.Module):
    """Text transfer baseline: classifier sees premise+hypothesis directly.

    Optionally truncates premise to max_tokens to simulate bandwidth constraint.
    """

    def __init__(self, backbone, hidden_dim=128, max_premise_tokens=None, is_lora=False):
        super().__init__()
        self.max_tokens = max_premise_tokens
        self.is_lora = is_lora
        if is_lora:
            self.backbone = backbone
        else:
            self._backbone = backbone
        h_size = backbone.hidden_size
        self.head = nn.Sequential(
            nn.Linear(h_size, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _truncate(self, text: str) -> str:
        if self.max_tokens is None:
            return text
        words = text.split()
        return " ".join(words[:self.max_tokens])

    def forward(self, premises, hypotheses, training=False):
        texts = [
            f"Context: {self._truncate(p)} Question: {h}"
            for p, h in zip(premises, hypotheses)
        ]
        backbone = self.backbone if self.is_lora else self._backbone
        if self.is_lora and training:
            h = backbone.encode_batch(texts).float()
        else:
            with torch.no_grad():
                h = backbone.encode_batch(texts).float()
        return self.head(h).squeeze(-1)


def build_ssr_system(model_id, device, comm_cfg, noise_std, lora_enc, lora_cls, lora_rank,
                     proj_dim=128, adapter_dim=128):
    if lora_enc:
        enc_bb = LoRALLM(model_id, device, "float16", lora_rank=lora_rank)
    else:
        enc_bb = FrozenLLM(model_id, device, "float16")
    if lora_cls:
        cls_bb = LoRALLM(model_id, device, "float16", lora_rank=lora_rank)
    else:
        cls_bb = FrozenLLM(model_id, device, "float16")

    comm = build_comm_channel(comm_cfg, input_dim=proj_dim)
    msg_dim = comm.message_dim()

    encoder = SSREncoder(
        enc_bb, ObsProjector(enc_bb.hidden_size, proj_dim, proj_dim),
        comm, noise_std, is_lora=lora_enc
    ).to(device)
    classifier = Classifier(
        cls_bb, ObsProjector(cls_bb.hidden_size, proj_dim, proj_dim),
        ReceiverAdapter(proj_dim, msg_dim, adapter_dim, adapter_dim),
        adapter_dim, is_lora=lora_cls
    ).to(device)
    return encoder, classifier


def train_epoch(model_or_pair, dataset, optimizer, device, batch_size=32, is_text=False):
    if is_text:
        model = model_or_pair
        model.train()
    else:
        enc, cls = model_or_pair
        enc.train()
        cls.train()

    total_loss = 0.0
    correct = 0
    total = 0
    n_batches = max(1, len(dataset) // batch_size)

    for _ in range(n_batches):
        batch = dataset.sample_batch(batch_size)
        contexts = [s.context for s in batch]
        questions = [s.question for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)

        if is_text:
            logits = model(contexts, questions, training=True)
        else:
            messages = enc(contexts, training=True)
            logits = cls(questions, messages, training=True)

        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad()
        loss.backward()
        if is_text:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        else:
            nn.utils.clip_grad_norm_(list(enc.parameters()) + list(cls.parameters()), 1.0)
        optimizer.step()

        total_loss += loss.item()
        correct += ((logits > 0).float() == labels).sum().item()
        total += len(batch)

    return {"train_loss": total_loss / n_batches, "train_acc": correct / total}


def evaluate(model_or_pair, dataset, device, batch_size=64, is_text=False):
    if is_text:
        model = model_or_pair
        model.eval()
    else:
        enc, cls = model_or_pair
        enc.eval()
        cls.eval()

    dataset.reset()
    correct = 0
    total = 0

    with torch.no_grad():
        for _ in range(max(1, len(dataset) // batch_size)):
            batch = dataset.get_batch(batch_size)
            contexts = [s.context for s in batch]
            questions = [s.question for s in batch]
            labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)

            if is_text:
                logits = model(contexts, questions, training=False)
            else:
                messages = enc(contexts, training=False)
                logits = cls(questions, messages, training=False)

            correct += ((logits > 0).float() == labels).sum().item()
            total += len(batch)

    return {"val_acc": correct / total}


def run_one(label, config, model_id, device, epochs, lr, seed, lora_rank, noise_std):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    torch.manual_seed(seed)
    np.random.seed(seed)
    train_data = NoisyRetrievalDataset("train", n_samples=15000, seed=seed)
    val_data = NoisyRetrievalDataset("val", n_samples=15000, seed=seed)
    print(f"  Train: {len(train_data)}, Val: {len(val_data)}")

    is_text = config["type"] == "text"
    t_start = time.time()

    if is_text:
        if config.get("lora", False):
            bb = LoRALLM(model_id, device, "float16", lora_rank=lora_rank)
        else:
            bb = FrozenLLM(model_id, device, "float16")
        model = TextTransferModel(
            bb, hidden_dim=128, max_premise_tokens=config.get("max_tokens"),
            is_lora=config.get("lora", False),
        ).to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        trainable = sum(p.numel() for p in params)
        print(f"  Trainable: {trainable:,} (text transfer, max_tokens={config.get('max_tokens')})")
        optimizer = torch.optim.Adam(params, lr=lr)
        model_or_pair = model
    else:
        comm_cfg = config["comm_cfg"]
        enc, cls = build_ssr_system(
            model_id, device, comm_cfg, noise_std,
            config.get("lora_enc", False), config.get("lora_cls", False), lora_rank,
        )
        params = [p for p in list(enc.parameters()) + list(cls.parameters()) if p.requires_grad]
        trainable = sum(p.numel() for p in params)
        print(f"  Trainable: {trainable:,} (comm={comm_cfg.type}, d={comm_cfg.dim}, noise={noise_std})")
        optimizer = torch.optim.Adam(params, lr=lr)
        model_or_pair = (enc, cls)

    history = {"train_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        tm = train_epoch(model_or_pair, train_data, optimizer, device, is_text=is_text)
        vm = evaluate(model_or_pair, val_data, device, is_text=is_text)
        history["train_loss"].append(tm["train_loss"])
        history["train_acc"].append(tm["train_acc"])
        history["val_acc"].append(vm["val_acc"])
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}: loss={tm['train_loss']:.4f} train={tm['train_acc']:.4f} val={vm['val_acc']:.4f} t={time.time()-t_start:.0f}s")

    final = evaluate(model_or_pair, val_data, device, is_text=is_text)
    elapsed = time.time() - t_start
    print(f"  FINAL: val_acc={final['val_acc']:.4f}, time={elapsed:.1f}s")

    del model_or_pair, optimizer
    torch.cuda.empty_cache()

    return {
        "label": label, "seed": seed,
        "final_val_acc": final["val_acc"],
        "time_seconds": elapsed,
        "config": {k: str(v) for k, v in config.items()},
        "history": history,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--noise", type=float, default=0.3, help="Gaussian noise std on comm channel")
    parser.add_argument("--seeds", nargs="*", type=int, default=[42, 123, 456])
    parser.add_argument("--output", default="results/noisy_retrieval.json")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    configs = [
        # Text transfer baselines
        ("text_full_frozen", {"type": "text", "max_tokens": None}),
        ("text_full_lora", {"type": "text", "max_tokens": None, "lora": True}),
        ("text_trunc16_frozen", {"type": "text", "max_tokens": 16}),
        ("text_trunc8_frozen", {"type": "text", "max_tokens": 8}),
        # SSR with d=32, noisy channel
        ("ssr_d32_frozen", {"type": "ssr", "comm_cfg": CommConfig(type="ssr", dim=32, normalize=False)}),
        ("ssr_d32_lora_enc", {"type": "ssr", "comm_cfg": CommConfig(type="ssr", dim=32, normalize=False), "lora_enc": True}),
        ("ssr_d32_lora_both", {"type": "ssr", "comm_cfg": CommConfig(type="ssr", dim=32, normalize=False), "lora_enc": True, "lora_cls": True}),
        # SSR with d=64, noisy channel
        ("ssr_d64_frozen", {"type": "ssr", "comm_cfg": CommConfig(type="ssr", dim=64, normalize=False)}),
        ("ssr_d64_lora_enc", {"type": "ssr", "comm_cfg": CommConfig(type="ssr", dim=64, normalize=False), "lora_enc": True}),
        # VQ-SSR with d=32, noisy channel
        ("vqssr_d32_frozen", {"type": "ssr", "comm_cfg": CommConfig(type="vq_ssr", dim=32, num_codes=32)}),
        ("vqssr_d32_lora_enc", {"type": "ssr", "comm_cfg": CommConfig(type="vq_ssr", dim=32, num_codes=32), "lora_enc": True}),
        # Discrete (no noise effect — already discrete)
        ("discrete_frozen", {"type": "ssr", "comm_cfg": CommConfig(type="discrete", dim=32, num_symbols=32)}),
        ("discrete_lora_enc", {"type": "ssr", "comm_cfg": CommConfig(type="discrete", dim=32, num_symbols=32), "lora_enc": True}),
        # No comm
        ("no_comm_frozen", {"type": "ssr", "comm_cfg": CommConfig(type="none", dim=1)}),
    ]

    all_results = []
    total_start = time.time()

    for name, config in configs:
        for seed in args.seeds:
            label = f"{name}_s{seed}"
            r = run_one(label, config, args.model, device, args.epochs, args.lr, seed,
                        args.lora_rank, args.noise)
            all_results.append(r)

    total_time = time.time() - total_start

    # Summary
    from collections import defaultdict
    by_name = defaultdict(list)
    for r in all_results:
        base = r["label"].rsplit("_s", 1)[0]
        by_name[base].append(r["final_val_acc"])

    print(f"\n{'='*60}")
    print(f"  NOISY RETRIEVAL RESULTS (noise_std={args.noise})")
    print(f"{'='*60}")
    print(f"{'Config':<30} {'Mean Acc':>10} {'Std':>8}")
    print("-" * 50)
    for name, accs in sorted(by_name.items(), key=lambda x: -np.mean(x[1])):
        print(f"{name:<30} {np.mean(accs):>8.4f} {np.std(accs):>8.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"config": vars(args), "results": all_results, "total_time": total_time}, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
