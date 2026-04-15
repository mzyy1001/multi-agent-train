#!/bin/bash
# Run focused noisy retrieval experiments — only the essential comparisons
# 6 configs x 3 seeds = 18 runs, 2 at a time (GPU limited)
set -e
cd ~/auto-research/multi-agent-train
export PYTHONPATH=.
PYTHON=/home/mzyy1001/miniconda3/envs/deepemu312/bin/python

MODEL="Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS=15
NOISE=0.3
LR=1e-4
LORA_RANK=8

mkdir -p results/noisy logs

run_one() {
    local label=$1 type=$2 comm=$3 dim=$4 lora_enc=$5 lora_cls=$6 max_tokens=$7 seed=$8
    local logfile="logs/noisy_${label}_s${seed}.log"

    echo "  Running: $label seed=$seed"

    $PYTHON -u -c "
import sys; sys.path.insert(0, '.')
import json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from src.backbone.llm import FrozenLLM
from src.backbone.lora_llm import LoRALLM
from src.comm import build_comm_channel
from src.config import CommConfig
from src.env_noisy_retrieval import NoisyRetrievalDataset
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter

torch.manual_seed($seed); np.random.seed($seed)
device = 'cuda'
train_data = NoisyRetrievalDataset('train', n_samples=15000, seed=$seed)
val_data = NoisyRetrievalDataset('val', n_samples=15000, seed=$seed)

if '$type' == 'text':
    # Text transfer
    if '$lora_enc' == 'True':
        bb = LoRALLM('$MODEL', device, 'float16', lora_rank=$LORA_RANK)
        is_lora = True
    else:
        bb = FrozenLLM('$MODEL', device, 'float16')
        is_lora = False
    h_size = bb.hidden_size
    max_tok = $max_tokens if $max_tokens > 0 else None

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_lora = is_lora
            if is_lora: self.backbone = bb
            else: self._backbone = bb
            self.head = nn.Sequential(nn.Linear(h_size, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
        def forward(self, premises, hypotheses, training=False):
            texts = [f'Context: {(\" \".join(p.split()[:max_tok]) if max_tok else p)} Question: {h}' for p, h in zip(premises, hypotheses)]
            backbone = self.backbone if self.is_lora else self._backbone
            if self.is_lora and training: h = backbone.encode_batch(texts).float()
            else:
                with torch.no_grad(): h = backbone.encode_batch(texts).float()
            return self.head(h).squeeze(-1)

    model = TextModel().to(device)
    params = [p for p in model.parameters() if p.requires_grad]

else:
    # SSR/VQ-SSR/Discrete with noisy channel
    comm_cfg = CommConfig(type='$comm', dim=$dim, normalize=False, num_codes=32, num_symbols=$dim)
    if '$lora_enc' == 'True':
        enc_bb = LoRALLM('$MODEL', device, 'float16', lora_rank=$LORA_RANK)
    else:
        enc_bb = FrozenLLM('$MODEL', device, 'float16')
    cls_bb = FrozenLLM('$MODEL', device, 'float16')

    comm = build_comm_channel(comm_cfg, input_dim=128)
    msg_dim = comm.message_dim()

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = ObsProjector(enc_bb.hidden_size, 128, 128)
            self.comm = comm
            self.noise_std = $NOISE
            self.is_lora = '$lora_enc' == 'True'
            if self.is_lora: self.backbone = enc_bb
            else: self._backbone = enc_bb
        def forward(self, texts, training=False):
            backbone = self.backbone if self.is_lora else self._backbone
            if self.is_lora and training: h = backbone.encode_batch(texts).float()
            else:
                with torch.no_grad(): h = backbone.encode_batch(texts).float()
            z = self.proj(h)
            m = self.comm(z)
            if self.training and self.noise_std > 0: m = m + torch.randn_like(m) * self.noise_std
            return m

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = ObsProjector(cls_bb.hidden_size, 128, 128)
            self.adapter = ReceiverAdapter(128, msg_dim, 128, 128)
            self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
            self._backbone = cls_bb
        def forward(self, texts, message, training=False):
            with torch.no_grad(): h = self._backbone.encode_batch(texts).float()
            z = self.proj(h)
            h_tilde = self.adapter(z, message)
            return self.head(h_tilde).squeeze(-1)

    enc = Encoder().to(device)
    cls = Classifier().to(device)
    params = [p for p in list(enc.parameters()) + list(cls.parameters()) if p.requires_grad]
    model = None

optimizer = torch.optim.Adam(params, lr=$LR)
t0 = time.time()

for epoch in range(1, ${EPOCHS}+1):
    if model: model.train()
    else: enc.train(); cls.train()
    loss_sum, correct, total = 0, 0, 0
    for _ in range(len(train_data)//32):
        batch = train_data.sample_batch(32)
        ctx = [s.context for s in batch]; q = [s.question for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
        if model: logits = model(ctx, q, training=True)
        else: logits = cls(q, enc(ctx, training=True), training=True)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
        loss_sum += loss.item(); correct += ((logits>0).float()==labels).sum().item(); total += len(batch)

    if epoch % 5 == 0 or epoch == 1:
        if model: model.eval()
        else: enc.eval(); cls.eval()
        val_data.reset(); vc, vt = 0, 0
        with torch.no_grad():
            for _ in range(len(val_data)//64):
                batch = val_data.get_batch(64)
                ctx = [s.context for s in batch]; q = [s.question for s in batch]
                labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
                if model: logits = model(ctx, q, training=False)
                else: logits = cls(q, enc(ctx, training=False), training=False)
                vc += ((logits>0).float()==labels).sum().item(); vt += len(batch)
        print(f'Epoch {epoch}: train={correct/total:.4f} val={vc/vt:.4f} loss={loss_sum*32/total:.4f} t={time.time()-t0:.0f}s', flush=True)

# Final eval
if model: model.eval()
else: enc.eval(); cls.eval()
val_data.reset(); vc, vt = 0, 0
with torch.no_grad():
    for _ in range(len(val_data)//64):
        batch = val_data.get_batch(64)
        ctx = [s.context for s in batch]; q = [s.question for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
        if model: logits = model(ctx, q, training=False)
        else: logits = cls(q, enc(ctx, training=False), training=False)
        vc += ((logits>0).float()==labels).sum().item(); vt += len(batch)
final_acc = vc/vt
elapsed = time.time()-t0
print(f'FINAL: val_acc={final_acc:.4f} time={elapsed:.1f}s', flush=True)

import json
with open('results/noisy/${label}_s${seed}.json', 'w') as f:
    json.dump({'label': '$label', 'seed': $seed, 'final_val_acc': final_acc, 'time': elapsed}, f)
" > "$logfile" 2>&1

    echo "  Done: $label seed=$seed ($(tail -1 $logfile))"
}

echo "=== Focused Noisy Retrieval Experiments ==="
echo "Noise std: $NOISE, Epochs: $EPOCHS"

# Run sequentially since GPU is near capacity
for seed in 42 123 456; do
    # 1. Text full (upper bound)
    run_one "text_full" "text" "" 0 "False" "False" 0 $seed

    # 2. Text truncated to 8 tokens (bandwidth-matched)
    run_one "text_trunc8" "text" "" 0 "False" "False" 8 $seed

    # 3. SSR d=32 frozen (through noise)
    run_one "ssr_d32" "ssr" "ssr" 32 "False" "False" 0 $seed

    # 4. SSR d=32 + LoRA encoder (gradient highway through noise)
    run_one "ssr_d32_lora" "ssr" "ssr" 32 "True" "False" 0 $seed

    # 5. Discrete frozen (no noise benefit)
    run_one "discrete_d32" "ssr" "discrete" 32 "False" "False" 0 $seed

    # 6. No comm (lower bound)
    run_one "nocomm" "ssr" "none" 1 "False" "False" 0 $seed
done

echo ""
echo "=== RESULTS ==="
$PYTHON -c "
import json, numpy as np
from pathlib import Path
from collections import defaultdict

by_name = defaultdict(list)
for f in sorted(Path('results/noisy').glob('*.json')):
    with open(f) as fh:
        r = json.load(fh)
        by_name[r['label']].append(r['final_val_acc'])

print(f\"{'Config':<25} {'Mean':>8} {'Std':>8}\")
print('-'*45)
for name in ['text_full', 'text_trunc8', 'ssr_d32', 'ssr_d32_lora', 'discrete_d32', 'nocomm']:
    if name in by_name:
        accs = by_name[name]
        print(f'{name:<25} {np.mean(accs):>8.4f} {np.std(accs):>8.4f}')
"
