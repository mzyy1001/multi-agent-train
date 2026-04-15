#!/bin/bash
# Novel domain experiment: test gradient highway on unfamiliar vocabulary
# where frozen LLM features are genuinely bad
set -e
cd ~/auto-research/multi-agent-train
export PYTHONPATH=.
PYTHON=/home/mzyy1001/miniconda3/envs/deepemu312/bin/python
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS=20
LR=1e-4
LORA_RANK=8
mkdir -p results/novel_domain logs

run_one() {
    local label=$1 seed=$2
    local logfile="logs/novel_${label}_s${seed}.log"
    echo "  Running: $label seed=$seed"

    $PYTHON -u -c "
import sys; sys.path.insert(0, '.')
import json, time, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from src.backbone.llm import FrozenLLM
from src.backbone.lora_llm import LoRALLM
from src.comm import build_comm_channel
from src.config import CommConfig
from src.env_novel_domain import NovelDomainDataset
from src.modules.obs_projector import ObsProjector
from src.modules.receiver_adapter import ReceiverAdapter

torch.manual_seed($seed); np.random.seed($seed)
device = 'cuda'
train = NovelDomainDataset('train', n_samples=15000, seed=$seed)
val = NovelDomainDataset('val', n_samples=15000, seed=$seed)
print(f'Train: {len(train)}, Val: {len(val)}', flush=True)

label = '$label'
is_text = 'text' in label
is_lora = 'lora' in label
is_discrete = 'discrete' in label

if is_text:
    if is_lora:
        bb = LoRALLM('$MODEL', device, 'float16', lora_rank=$LORA_RANK); is_lora_bb = True
    else:
        bb = FrozenLLM('$MODEL', device, 'float16'); is_lora_bb = False
    h_size = bb.hidden_size
    max_tok = 8 if 'trunc' in label else None

    class TextModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.is_lora = is_lora_bb
            if self.is_lora: self.backbone = bb
            else: self._backbone = bb
            self.head = nn.Sequential(nn.Linear(h_size, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
        def forward(self, contexts, questions, candidates, training=False):
            texts = [f'Context: {\" \".join(c.split()[:max_tok]) if max_tok else c} Question: {q} Candidate: {a}'
                     for c, q, a in zip(contexts, questions, candidates)]
            backbone = self.backbone if self.is_lora else self._backbone
            if self.is_lora and training: h = backbone.encode_batch(texts).float()
            else:
                with torch.no_grad(): h = backbone.encode_batch(texts).float()
            return self.head(h).squeeze(-1)
    model = TextModel().to(device)
    params = [p for p in model.parameters() if p.requires_grad]
else:
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
            if is_lora: self.backbone = enc_bb
            else: self._backbone = enc_bb
        def forward(self, texts, training=False):
            backbone = self.backbone if self.is_lora else self._backbone
            if self.is_lora and training: h = backbone.encode_batch(texts).float()
            else:
                with torch.no_grad(): h = backbone.encode_batch(texts).float()
            return self.comm(self.proj(h))
    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = ObsProjector(cls_bb.hidden_size, 128, 128)
            self.adapter = ReceiverAdapter(128, msg_dim, 128, 128)
            self.head = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
            self._backbone = cls_bb
        def forward(self, questions, candidates, message, training=False):
            texts = [f'Question: {q} Candidate: {a}' for q, a in zip(questions, candidates)]
            with torch.no_grad(): h = self._backbone.encode_batch(texts).float()
            return self.head(self.adapter(self.proj(h), message)).squeeze(-1)
    enc = Encoder().to(device); cls = Classifier().to(device)
    params = [p for p in list(enc.parameters()) + list(cls.parameters()) if p.requires_grad]
    model = None

trainable = sum(p.numel() for p in params)
print(f'Trainable: {trainable:,}', flush=True)
optimizer = torch.optim.Adam(params, lr=$LR)
t0 = time.time()

for epoch in range(1, ${EPOCHS}+1):
    if model: model.train()
    else: enc.train(); cls.train()
    correct, total = 0, 0
    for _ in range(len(train)//32):
        batch = train.sample_batch(32)
        ctx = [s.context for s in batch]; q = [s.question for s in batch]
        cand = [s.candidate for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
        if model: logits = model(ctx, q, cand, training=True)
        else: logits = cls(q, cand, enc(ctx, training=True), training=True)
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
                ctx = [s.context for s in batch]; q = [s.question for s in batch]
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
with open('results/novel_domain/${label}_s${seed}.json', 'w') as f:
    json.dump({'label': '$label', 'seed': $seed, 'final_val_acc': vc/vt, 'time': time.time()-t0}, f)
" > "$logfile" 2>&1
    echo "  Done: $(grep FINAL $logfile)"
}

echo "=== Novel Domain: Gradient Highway on Unfamiliar Vocabulary ==="
echo "LLM has never seen these words — frozen features should be poor"
echo "LoRA adaptation through gradient highway should be essential"
echo ""

for seed in 42 123 456; do
    # Text baselines
    run_one "text_full" $seed
    run_one "text_full_lora" $seed
    run_one "text_trunc8" $seed
    run_one "text_trunc8_lora" $seed
    # SSR
    run_one "ssr_d32" $seed
    run_one "ssr_d32_lora" $seed
    # Discrete
    run_one "discrete_d32" $seed
    # No comm
    run_one "nocomm" $seed
done

echo ""
echo "=== NOVEL DOMAIN RESULTS ==="
$PYTHON -c "
import json, numpy as np
from pathlib import Path
from collections import defaultdict

by_name = defaultdict(list)
for f in sorted(Path('results/novel_domain').glob('*.json')):
    with open(f) as fh:
        r = json.load(fh)
        by_name[r['label']].append(r['final_val_acc'])

print(f\"{'Config':<25} {'Mean':>8} {'Std':>8} {'N':>4}\")
print('-'*48)
for name in ['text_full', 'text_full_lora', 'text_trunc8', 'text_trunc8_lora',
             'ssr_d32', 'ssr_d32_lora', 'discrete_d32', 'nocomm']:
    if name in by_name:
        accs = by_name[name]
        print(f'{name:<25} {np.mean(accs):>8.4f} {np.std(accs):>8.4f} {len(accs):>4}')

# Key comparison
if 'ssr_d32_lora' in by_name and 'ssr_d32' in by_name:
    lora_m = np.mean(by_name['ssr_d32_lora'])
    frozen_m = np.mean(by_name['ssr_d32'])
    print(f'\nGradient highway delta: {lora_m-frozen_m:+.4f} (SSR+LoRA vs SSR frozen)')
if 'text_trunc8_lora' in by_name and 'text_trunc8' in by_name:
    tl = np.mean(by_name['text_trunc8_lora'])
    tf = np.mean(by_name['text_trunc8'])
    print(f'Text LoRA delta: {tl-tf:+.4f} (text+LoRA vs text frozen)')
if 'ssr_d32_lora' in by_name and 'text_trunc8' in by_name:
    sl = np.mean(by_name['ssr_d32_lora'])
    tf = np.mean(by_name['text_trunc8'])
    print(f'SSR+LoRA vs text_trunc8: {sl-tf:+.4f}')
"
