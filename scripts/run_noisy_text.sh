#!/bin/bash
# Run text transfer baselines WITH noise on the text (token dropout/corruption)
# This makes the comparison fair: SSR goes through noise, text should too
set -e
cd ~/auto-research/multi-agent-train
export PYTHONPATH=.
PYTHON=/home/mzyy1001/miniconda3/envs/deepemu312/bin/python
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
EPOCHS=15
LR=1e-4
mkdir -p results/noisy_text logs

# For text noise: randomly drop/replace words in the premise before passing to classifier
# This simulates token-level noise on a text channel

for seed in 42 123 456; do
for noise_rate in 0.0 0.1 0.3 0.5; do
    label="text_noisy_${noise_rate}_s${seed}"
    logfile="logs/${label}.log"
    echo "Running: $label"

    $PYTHON -u -c "
import sys; sys.path.insert(0, '.')
import json, time, random, torch, numpy as np
import torch.nn as nn, torch.nn.functional as F
from src.backbone.llm import FrozenLLM
from src.env_noisy_retrieval import NoisyRetrievalDataset

torch.manual_seed($seed); np.random.seed($seed); random.seed($seed)
device = 'cuda'
train_data = NoisyRetrievalDataset('train', n_samples=15000, seed=$seed)
val_data = NoisyRetrievalDataset('val', n_samples=15000, seed=$seed)

bb = FrozenLLM('$MODEL', device, 'float16')
h_size = bb.hidden_size

def corrupt_text(text, rate):
    if rate <= 0: return text
    words = text.split()
    result = []
    vocab = ['the', 'a', 'is', 'in', 'at', 'on', 'to', 'of', 'and', 'for',
             'red', 'blue', 'green', 'big', 'small', 'old', 'new', 'fast', 'slow', 'hot']
    for w in words:
        r = random.random()
        if r < rate * 0.5:
            continue  # drop word
        elif r < rate:
            result.append(random.choice(vocab))  # replace with random word
        else:
            result.append(w)
    return ' '.join(result) if result else text

class TextModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._backbone = bb
        self.head = nn.Sequential(nn.Linear(h_size, 128), nn.GELU(), nn.Linear(128, 128), nn.GELU(), nn.Linear(128, 1))
    def forward(self, premises, hypotheses, training=False):
        noisy_premises = [corrupt_text(p, $noise_rate) for p in premises]
        texts = [f'Context: {\" \".join(p.split()[:8])} Question: {h}' for p, h in zip(noisy_premises, hypotheses)]
        with torch.no_grad(): h = self._backbone.encode_batch(texts).float()
        return self.head(h).squeeze(-1)

model = TextModel().to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.Adam(params, lr=$LR)
t0 = time.time()

for epoch in range(1, ${EPOCHS}+1):
    model.train(); loss_sum, correct, total = 0, 0, 0
    for _ in range(len(train_data)//32):
        batch = train_data.sample_batch(32)
        ctx = [s.context for s in batch]; q = [s.question for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
        logits = model(ctx, q, training=True)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0); optimizer.step()
        loss_sum += loss.item(); correct += ((logits>0).float()==labels).sum().item(); total += len(batch)
    if epoch % 5 == 0 or epoch == 1:
        model.eval(); val_data.reset(); vc, vt = 0, 0
        with torch.no_grad():
            for _ in range(len(val_data)//64):
                batch = val_data.get_batch(64)
                ctx = [s.context for s in batch]; q = [s.question for s in batch]
                labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
                logits = model(ctx, q, training=False)
                vc += ((logits>0).float()==labels).sum().item(); vt += len(batch)
        print(f'Epoch {epoch}: train={correct/total:.4f} val={vc/vt:.4f} t={time.time()-t0:.0f}s', flush=True)

model.eval(); val_data.reset(); vc, vt = 0, 0
with torch.no_grad():
    for _ in range(len(val_data)//64):
        batch = val_data.get_batch(64)
        ctx = [s.context for s in batch]; q = [s.question for s in batch]
        labels = torch.tensor([s.label for s in batch], dtype=torch.float32, device=device)
        logits = model(ctx, q, training=False)
        vc += ((logits>0).float()==labels).sum().item(); vt += len(batch)
print(f'FINAL: val_acc={vc/vt:.4f} time={time.time()-t0:.1f}s', flush=True)
with open('results/noisy_text/${label}.json', 'w') as f:
    json.dump({'label': 'text_noisy_trunc8', 'noise_rate': $noise_rate, 'seed': $seed, 'final_val_acc': vc/vt}, f)
" > "$logfile" 2>&1

    echo "  Done: $(grep FINAL $logfile)"
done
done

echo ""
echo "=== NOISY TEXT RESULTS ==="
$PYTHON -c "
import json, numpy as np
from pathlib import Path
from collections import defaultdict

by_noise = defaultdict(list)
for f in sorted(Path('results/noisy_text').glob('*.json')):
    with open(f) as fh:
        r = json.load(fh)
        by_noise[r['noise_rate']].append(r['final_val_acc'])

print(f\"{'Noise Rate':<15} {'Mean Acc':>10} {'Std':>8} {'N':>5}\")
print('-'*40)
for noise in sorted(by_noise.keys()):
    accs = by_noise[noise]
    print(f'{noise:<15} {np.mean(accs):>10.4f} {np.std(accs):>8.4f} {len(accs):>5}')
"
