#!/bin/bash
# Heterogeneous model experiment: SmolLM2-135M encoder + Qwen2.5-0.5B classifier
# Tests whether the gradient highway bridges different architectures
set -e
cd ~/auto-research/multi-agent-train
export PYTHONPATH=.
PYTHON=/home/mzyy1001/miniconda3/envs/deepemu312/bin/python
ENC_MODEL="HuggingFaceTB/SmolLM2-135M-Instruct"
CLS_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS=20
LR=1e-4
LORA_RANK=8
mkdir -p results/heterogeneous logs

run_one() {
    local label=$1 seed=$2
    local logfile="logs/hetero_${label}_s${seed}.log"
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
is_lora = 'lora' in label
is_text = 'text' in label
is_nocomm = 'nocomm' in label
is_homo_qwen = 'homo_qwen' in label
is_homo_smol = 'homo_smol' in label

# Determine models
if is_homo_qwen:
    enc_model_id = '$CLS_MODEL'  # Qwen for both
    cls_model_id = '$CLS_MODEL'
elif is_homo_smol:
    enc_model_id = '$ENC_MODEL'  # SmolLM for both
    cls_model_id = '$ENC_MODEL'
else:
    enc_model_id = '$ENC_MODEL'  # Heterogeneous: SmolLM enc + Qwen cls
    cls_model_id = '$CLS_MODEL'

print(f'Encoder: {enc_model_id}', flush=True)
print(f'Classifier: {cls_model_id}', flush=True)

if is_nocomm:
    # No communication: classifier sees question+candidate only, no context info
    cls_bb_nc = FrozenLLM(cls_model_id, device, 'float16')
    h_size_nc = cls_bb_nc.hidden_size
    class NoCommModel(nn.Module):
        def __init__(self):
            super().__init__()
            self._backbone = cls_bb_nc
            self.head = nn.Sequential(nn.Linear(h_size_nc, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
        def forward(self, contexts, questions, candidates, training=False):
            texts = [f'Question: {q} Candidate: {a}' for q, a in zip(questions, candidates)]
            with torch.no_grad(): h = self._backbone.encode_batch(texts).float()
            return self.head(h).squeeze(-1)
    model = NoCommModel().to(device)
    params = [p for p in model.parameters() if p.requires_grad]

elif is_text:
    # Text transfer uses classifier model only (sees both context + question)
    if is_lora:
        bb = LoRALLM(cls_model_id, device, 'float16', lora_rank=$LORA_RANK); is_lora_bb = True
    else:
        bb = FrozenLLM(cls_model_id, device, 'float16'); is_lora_bb = False
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
    # SSR: encoder and classifier have DIFFERENT models
    comm_cfg = CommConfig(type='ssr', dim=32, normalize=False)
    if is_lora:
        enc_bb = LoRALLM(enc_model_id, device, 'float16', lora_rank=$LORA_RANK)
    else:
        enc_bb = FrozenLLM(enc_model_id, device, 'float16')
    cls_bb = FrozenLLM(cls_model_id, device, 'float16')
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
        def forward(self, questions, candidates, message):
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
        else: logits = cls(q, cand, enc(ctx, training=True))
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
                else: logits = cls(q, cand, enc(ctx, training=False))
                vc += ((logits>0).float()==labels).sum().item(); vt += len(batch)
        print(f'Epoch {epoch}: train={correct/total:.4f} val={vc/vt:.4f} t={time.time()-t0:.0f}s', flush=True)

if model: model.eval()
else: enc.eval(); cls.eval()
val.reset(); vc, vt = 0, 0
with torch.no_grad():
    for _ in range(len(val)//64):
        batch = val.get_batch(64); ctx = [s.context for s in batch]
        q = [s.question for s in batch]; cand = [s.candidate for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
        if model: logits = model(ctx, q, cand, training=False)
        else: logits = cls(q, cand, enc(ctx, training=False))
        vc += ((logits>0).float()==labels).sum().item(); vt += len(batch)
print(f'FINAL: val_acc={vc/vt:.4f} time={time.time()-t0:.1f}s', flush=True)
with open('results/heterogeneous/${label}_s${seed}.json', 'w') as f:
    json.dump({'label': '$label', 'seed': $seed, 'final_val_acc': vc/vt, 'time': time.time()-t0}, f)
" > "$logfile" 2>&1
    echo "  Done: $(grep FINAL $logfile)"
}

echo "=== Heterogeneous Model Experiment ==="
echo "Encoder: SmolLM2-135M (hidden=576), Classifier: Qwen2.5-0.5B (hidden=896)"
echo ""

for seed in 42 123 456; do
    # === Heterogeneous: SmolLM encoder + Qwen classifier ===
    run_one "hetero_ssr" $seed              # SSR frozen (no gradient highway)
    run_one "hetero_ssr_lora" $seed         # SSR + LoRA on SmolLM encoder (gradient highway!)
    run_one "hetero_text_trunc8" $seed      # Text 8tok, Qwen only, frozen
    run_one "hetero_text_trunc8_lora" $seed # Text 8tok, Qwen + LoRA on classifier
    run_one "hetero_nocomm" $seed           # No communication (lower bound)

    # === Homogeneous baselines ===
    run_one "homo_qwen_ssr_lora" $seed      # Qwen-Qwen SSR+LoRA
    run_one "homo_smol_ssr_lora" $seed      # SmolLM-SmolLM SSR+LoRA
done

echo ""
echo "=== HETEROGENEOUS RESULTS ==="
$PYTHON -c "
import json, numpy as np
from pathlib import Path
from collections import defaultdict

by_name = defaultdict(list)
for f in sorted(Path('results/heterogeneous').glob('*.json')):
    with open(f) as fh:
        r = json.load(fh)
        by_name[r['label']].append(r['final_val_acc'])

print(f\"{'Config':<30} {'Mean':>8} {'Std':>8} {'N':>4}\")
print('-'*55)
for name in ['homo_qwen_ssr_lora', 'hetero_ssr_lora', 'homo_smol_ssr_lora',
             'hetero_text_trunc8_lora', 'hetero_text_trunc8',
             'hetero_ssr', 'hetero_nocomm']:
    if name in by_name:
        accs = by_name[name]
        print(f'{name:<30} {np.mean(accs):>8.4f} {np.std(accs):>8.4f} {len(accs):>4}')

if 'hetero_ssr_lora' in by_name and 'hetero_ssr' in by_name:
    delta = np.mean(by_name['hetero_ssr_lora']) - np.mean(by_name['hetero_ssr'])
    print(f'\nHeterogeneous gradient highway delta: {delta:+.4f}')
"
