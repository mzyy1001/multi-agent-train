#!/bin/bash
# Noise sweep: test SSR+LoRA vs discrete vs text_trunc8 at multiple noise levels
# Also tests d=64 and VQ-SSR at noise=0.3
set -e
cd ~/auto-research/multi-agent-train
export PYTHONPATH=.
PYTHON=/home/mzyy1001/miniconda3/envs/deepemu312/bin/python
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS=15
LR=1e-4
LORA_RANK=8
mkdir -p results/noise_sweep logs

run_one() {
    local label=$1 type=$2 comm=$3 dim=$4 lora_enc=$5 noise=$6 max_tokens=$7 seed=$8
    local logfile="logs/sweep_${label}_n${noise}_s${seed}.log"
    echo "  Running: $label noise=$noise seed=$seed"

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
    if '$lora_enc' == 'True':
        bb = LoRALLM('$MODEL', device, 'float16', lora_rank=$LORA_RANK); is_lora = True
    else:
        bb = FrozenLLM('$MODEL', device, 'float16'); is_lora = False
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
            self.comm = comm; self.noise_std = $noise
            self.is_lora = '$lora_enc' == 'True'
            if self.is_lora: self.backbone = enc_bb
            else: self._backbone = enc_bb
        def forward(self, texts, training=False):
            backbone = self.backbone if self.is_lora else self._backbone
            if self.is_lora and training: h = backbone.encode_batch(texts).float()
            else:
                with torch.no_grad(): h = backbone.encode_batch(texts).float()
            z = self.proj(h); m = self.comm(z)
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
            z = self.proj(h); h_tilde = self.adapter(z, message)
            return self.head(h_tilde).squeeze(-1)
    enc = Encoder().to(device); cls = Classifier().to(device)
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
        print(f'Epoch {epoch}: train={correct/total:.4f} val={vc/vt:.4f} t={time.time()-t0:.0f}s', flush=True)
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
print(f'FINAL: val_acc={vc/vt:.4f} time={time.time()-t0:.1f}s', flush=True)
with open('results/noise_sweep/${label}_n${noise}_s${seed}.json', 'w') as f:
    json.dump({'label': '$label', 'noise': $noise, 'seed': $seed, 'final_val_acc': vc/vt}, f)
" > "$logfile" 2>&1
    echo "  Done: $label noise=$noise seed=$seed ($(grep FINAL $logfile))"
}

echo "=== Noise Sweep + Ablations ==="

# Use seed 42 only for sweep (faster), 3 seeds for key comparisons
SEED=42

# 1. Noise sweep: σ=0.0, 0.1, 0.5 for key methods
for noise in 0.0 0.1 0.5; do
    run_one "ssr_d32_lora" "ssr" "ssr" 32 "True" "$noise" 0 $SEED
    run_one "ssr_d32" "ssr" "ssr" 32 "False" "$noise" 0 $SEED
    run_one "discrete_d32" "ssr" "discrete" 32 "False" "$noise" 0 $SEED
    run_one "text_trunc8" "text" "" 0 "False" "$noise" 8 $SEED
done

# 2. d=64 SSR at noise=0.3
run_one "ssr_d64" "ssr" "ssr" 64 "False" "0.3" 0 $SEED
run_one "ssr_d64_lora" "ssr" "ssr" 64 "True" "0.3" 0 $SEED

# 3. VQ-SSR at noise=0.3
run_one "vqssr_d32" "ssr" "vq_ssr" 32 "False" "0.3" 0 $SEED
run_one "vqssr_d32_lora" "ssr" "vq_ssr" 32 "True" "0.3" 0 $SEED

echo ""
echo "=== NOISE SWEEP RESULTS ==="
$PYTHON -c "
import json, numpy as np
from pathlib import Path
from collections import defaultdict

results = []
for f in sorted(Path('results/noise_sweep').glob('*.json')):
    with open(f) as fh:
        results.append(json.load(fh))

# Group by noise level
by_noise = defaultdict(dict)
for r in results:
    by_noise[r['noise']][r['label']] = r['final_val_acc']

print(f\"{'Method':<20} {'σ=0.0':>8} {'σ=0.1':>8} {'σ=0.3':>8} {'σ=0.5':>8}\")
print('-'*50)
for method in ['text_trunc8', 'ssr_d32_lora', 'ssr_d32', 'discrete_d32']:
    vals = []
    for noise in [0.0, 0.1, 0.3, 0.5]:
        v = by_noise.get(noise, {}).get(method, None)
        vals.append(f'{v:.4f}' if v else '  --  ')
    # Add σ=0.3 from main results if not in sweep
    print(f'{method:<20} {\"  \".join(vals)}')

print()
print('Dimension ablation (σ=0.3):')
for method in ['ssr_d32', 'ssr_d32_lora', 'ssr_d64', 'ssr_d64_lora', 'vqssr_d32', 'vqssr_d32_lora']:
    v = by_noise.get(0.3, {}).get(method, None)
    if v: print(f'  {method:<20} {v:.4f}')
"
