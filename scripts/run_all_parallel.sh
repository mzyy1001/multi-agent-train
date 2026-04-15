#!/bin/bash
# Run all 21 frozen-backbone experiments in parallel batches
# Each batch loads the backbone once per process (~500MB each)
# With RTX 4090 (24GB) and 2 other experiments running (~11GB used),
# we can run ~4-5 at a time safely

cd ~/auto-research/multi-agent-train
export PYTHONPATH=.
mkdir -p results/individual logs

EPISODES=10000
DEVICE=cuda
CONDA_ENV=deepemu312
MAX_PARALLEL=4

methods=(ssr ssr_no_ln ssr_v2 vq_ssr discrete continuous no_comm)
seeds=(42 123 456)

# Build list of all jobs
jobs=()
for method in "${methods[@]}"; do
    for seed in "${seeds[@]}"; do
        jobs+=("${method}:${seed}")
    done
done

total=${#jobs[@]}
echo "Total jobs: $total, max parallel: $MAX_PARALLEL"

# Run in batches
idx=0
while [ $idx -lt $total ]; do
    pids=()
    batch_end=$((idx + MAX_PARALLEL))
    if [ $batch_end -gt $total ]; then
        batch_end=$total
    fi

    echo ""
    echo "=== Batch $((idx/MAX_PARALLEL + 1)): jobs $((idx+1))-${batch_end} of $total ==="

    for ((i=idx; i<batch_end; i++)); do
        IFS=':' read -r method seed <<< "${jobs[$i]}"
        logfile="logs/${method}_s${seed}.log"
        echo "  Starting: $method seed=$seed -> $logfile"
        conda run -n $CONDA_ENV bash -c "cd ~/auto-research/multi-agent-train && PYTHONPATH=. python scripts/run_parallel.py --method $method --seed $seed --episodes $EPISODES --device $DEVICE" > "$logfile" 2>&1 &
        pids+=($!)
    done

    # Wait for all in this batch
    echo "  Waiting for ${#pids[@]} processes: ${pids[*]}"
    for pid in "${pids[@]}"; do
        wait $pid
        status=$?
        echo "  PID $pid finished (exit=$status)"
    done

    idx=$batch_end
done

echo ""
echo "=== All $total jobs complete ==="

# Aggregate results
python3 -c "
import json, os
from pathlib import Path

results = []
for f in sorted(Path('results/individual').glob('*.json')):
    with open(f) as fh:
        results.append(json.load(fh))

# Group by method
from collections import defaultdict
import numpy as np

by_method = defaultdict(list)
for r in results:
    by_method[r['method']].append(r)

aggregated = []
for method, runs in by_method.items():
    rewards = [r['final_eval_reward'] for r in runs]
    aggregated.append({
        'method': method,
        'seeds': [r['seed'] for r in runs],
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'per_seed': runs,
    })

aggregated.sort(key=lambda x: x['mean_reward'], reverse=True)

save_data = {
    'config': {'episodes': $EPISODES, 'seeds': [42, 123, 456], 'comm_dim': 8},
    'aggregated': aggregated,
}

with open('results/comparison_v2.json', 'w') as f:
    json.dump(save_data, f, indent=2)

print()
print('RESULTS SUMMARY')
print('='*60)
print(f\"{'Method':<20} {'Mean':>10} {'Std':>10} {'Range':>20}\")
print('-'*60)
for a in aggregated:
    print(f\"{a['method']:<20} {a['mean_reward']:>10.2f} {a['std_reward']:>10.2f} [{a['min_reward']:.1f}, {a['max_reward']:.1f}]\")
"
