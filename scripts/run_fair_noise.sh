#!/bin/bash
# Fair noise comparison: noise applied at hidden state level for ALL methods
# This eliminates the asymmetric noise model criticism
set -e
cd ~/auto-research/multi-agent-train
export PYTHONPATH=.
PYTHON=/home/mzyy1001/miniconda3/envs/deepemu312/bin/python
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS=15
LR=1e-4
LORA_RANK=8
mkdir -p results/fair_noise logs

run_one() {
    local label=$1 seed=$2 noise=$3
    local logfile="logs/fair_${label}_n${noise}_s${seed}.log"
    echo "  Running: $label noise=$noise seed=$seed"

    $PYTHON -u -c "
import sys; sys.path.insert(0, '.')
import json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from src.backbone.llm import FrozenLLM
from src.backbone.lora_llm import LoRALLM
from src.comm import build_comm_channel
from src.config import CommConfig
from src.env_noisy_retrieval_v2 import NoisyRetrievalDatasetV2
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter

torch.manual_seed($seed); np.random.seed($seed)
device = 'cuda'
train = NoisyRetrievalDatasetV2('train', n_samples=15000, seed=$seed)
val = NoisyRetrievalDatasetV2('val', n_samples=15000, seed=$seed)
print(f'Train: {len(train)}, Val: {len(val)}', flush=True)

label = '$label'
noise_std = $noise

# Determine config from label
is_text = 'text' in label
is_lora = 'lora' in label
is_discrete = 'discrete' in label

if is_text:
    # Text transfer: encode context+question+candidate together
    # Noise applied to encoder hidden state
    if is_lora:
        bb = LoRALLM('$MODEL', device, 'float16', lora_rank=$LORA_RANK)
        is_lora_bb = True
    else:
        bb = FrozenLLM('$MODEL', device, 'float16')
        is_lora_bb = False

    h_size = bb.hidden_size

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_lora = is_lora_bb
            if self.is_lora:
                self.backbone = bb
            else:
                self._backbone = bb
            self.head = nn.Sequential(
                nn.Linear(h_size, 128), nn.GELU(),
                nn.Linear(128, 128), nn.GELU(),
                nn.Linear(128, 1))

        def forward(self, contexts, questions, candidates, training=False):
            # Truncate context to 8 words for bandwidth matching
            texts = [f'Context: {\" \".join(c.split()[:8])} Question: {q} Candidate: {a}'
                     for c, q, a in zip(contexts, questions, candidates)]
            backbone = self.backbone if self.is_lora else self._backbone
            if self.is_lora and training:
                h = backbone.encode_batch(texts).float()
            else:
                with torch.no_grad():
                    h = backbone.encode_batch(texts).float()
            # Apply SAME Gaussian noise to hidden state
            if self.training and noise_std > 0:
                h = h + torch.randn_like(h) * noise_std
            return self.head(h).squeeze(-1)

    model = TextModel().to(device)
    params = [p for p in model.parameters() if p.requires_grad]

else:
    # SSR or discrete: encoder sees context, classifier sees question+candidate
    if is_discrete:
        comm_cfg = CommConfig(type='discrete', dim=32, num_symbols=32)
    else:
        comm_cfg = CommConfig(type='ssr', dim=32, normalize=False)

    if is_lora:
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
            self.is_lora = is_lora
            if is_lora:
                self.backbone = enc_bb
            else:
                self._backbone = enc_bb
        def forward(self, texts, training=False):
            backbone = self.backbone if self.is_lora else self._backbone
            if self.is_lora and training:
                h = backbone.encode_batch(texts).float()
            else:
                with torch.no_grad():
                    h = backbone.encode_batch(texts).float()
            # Apply noise to hidden state (SAME as text model)
            if self.training and noise_std > 0:
                h = h + torch.randn_like(h) * noise_std
            z = self.proj(h)
            return self.comm(z)

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = ObsProjector(cls_bb.hidden_size, 128, 128)
            self.adapter = ReceiverAdapter(128, msg_dim, 128, 128)
            self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
            self._backbone = cls_bb
        def forward(self, questions, candidates, message, training=False):
            texts = [f'Question: {q} Candidate: {a}' for q, a in zip(questions, candidates)]
            with torch.no_grad():
                h = self._backbone.encode_batch(texts).float()
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
    correct, total = 0, 0
    for _ in range(len(train)//32):
        batch = train.sample_batch(32)
        ctx = [s.context for s in batch]
        q = [s.question for s in batch]
        cand = [s.candidate for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
        if model:
            logits = model(ctx, q, cand, training=True)
        else:
            logits = cls(q, cand, enc(ctx, training=True), training=True)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
        correct += ((logits>0).float()==labels).sum().item(); total += len(batch)
    if epoch % 5 == 0 or epoch == 1:
        if model: model.eval()
        else: enc.eval(); cls.eval()
        val.reset(); vc, vt = 0, 0
        with torch.no_grad():
            for _ in range(len(val)//64):
                batch = val.get_batch(64)
                ctx = [s.context for s in batch]
                q = [s.question for s in batch]
                cand = [s.candidate for s in batch]
                labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
                if model: logits = model(ctx, q, cand, training=False)
                else: logits = cls(q, cand, enc(ctx, training=False), training=False)
                vc += ((logits>0).float()==labels).sum().item(); vt += len(batch)
        print(f'Epoch {epoch}: train={correct/total:.4f} val={vc/vt:.4f} t={time.time()-t0:.0f}s', flush=True)

if model: model.eval()
else: enc.eval(); cls.eval()
val.reset(); vc, vt = 0, 0
with torch.no_grad():
    for _ in range(len(val)//64):
        batch = val.get_batch(64)
        ctx = [s.context for s in batch]; q = [s.question for s in batch]
        cand = [s.candidate for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
        if model: logits = model(ctx, q, cand, training=False)
        else: logits = cls(q, cand, enc(ctx, training=False), training=False)
        vc += ((logits>0).float()==labels).sum().item(); vt += len(batch)
print(f'FINAL: val_acc={vc/vt:.4f} time={time.time()-t0:.1f}s', flush=True)
with open('results/fair_noise/${label}_n${noise}_s${seed}.json', 'w') as f:
    json.dump({'label': '$label', 'noise': $noise, 'seed': $seed, 'final_val_acc': vc/vt}, f)
" > "$logfile" 2>&1
    echo "  Done: $(grep FINAL $logfile)"
}

echo "=== Fair Noise Comparison ==="
echo "Noise applied at hidden state level for ALL methods"

for seed in 42 123 456; do
for noise in 0.0 0.3 0.5 1.0; do
    run_one "text_trunc8" $seed $noise
    run_one "ssr_d32" $seed $noise
    run_one "ssr_d32_lora" $seed $noise
    run_one "discrete_d32" $seed $noise
done
done

echo ""
echo "=== FAIR NOISE RESULTS ==="
$PYTHON -c "
import json, numpy as np
from pathlib import Path
from collections import defaultdict

by_config = defaultdict(lambda: defaultdict(list))
for f in sorted(Path('results/fair_noise').glob('*.json')):
    with open(f) as fh:
        r = json.load(fh)
        by_config[r['label']][r['noise']].append(r['final_val_acc'])

print(f\"{'Method':<20} {'σ=0.0':>8} {'σ=0.3':>8} {'σ=0.5':>8} {'σ=1.0':>8}\")
print('-'*55)
for method in ['text_trunc8', 'ssr_d32_lora', 'ssr_d32', 'discrete_d32']:
    vals = []
    for noise in [0.0, 0.3, 0.5, 1.0]:
        accs = by_config[method].get(noise, [])
        if accs:
            vals.append(f'{np.mean(accs):.4f}')
        else:
            vals.append('  --  ')
    print(f'{method:<20} {\"  \".join(vals)}')
"
