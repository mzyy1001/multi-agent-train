# Improved Experiment Plan

## Problem Statement

SSR (Structured Semantic Representation) underperforms discrete Gumbel-Softmax communication
(-66.02 vs -45.32) in the original 5K-episode single-seed comparison. This plan addresses
the gap through architectural improvements, proper ablations, and rigorous evaluation.

## Root Cause Analysis

1. **Task-structure mismatch**: Speaker-Listener has 3 categorical landmarks — discrete
   communication naturally aligns with this structure
2. **LayerNorm removes magnitude**: Normalizing the message strips useful signal strength
3. **Insufficient training**: 5K episodes (10% of intended 50K) — SSR's deeper bottleneck
   may need more samples to develop structured representations
4. **No progressive refinement**: Discrete has tau annealing; SSR has no equivalent
5. **Single seed**: Results may not be robust

## New Methods

### VQ-SSR (Vector-Quantized SSR)
- Combines SSR's differentiable MLP bottleneck with vector quantization
- Learns a codebook of discrete message prototypes
- Straight-through estimator preserves gradient flow during training
- At eval: messages snap to nearest codebook entry (discrete, interpretable)
- Bridges the gap between continuous differentiability and discrete structure

### SSR v2 (Improved SSR)
- Removes LayerNorm by default (preserves magnitude information)
- Adds residual connection from input projection
- Configurable expansion factor and optional dropout

### SSR No-LN (Ablation)
- Original SSR architecture with LayerNorm disabled
- Isolates the effect of normalization

## Experiment Grid

### Phase 1: Ablation Study (10K episodes, 3 seeds)
| Method | Description | Claim tested |
|--------|-------------|-------------|
| ssr | Original SSR (MLP + LayerNorm) | Baseline proposed method |
| ssr_no_ln | SSR without LayerNorm | LayerNorm hurts performance |
| ssr_v2 | SSR + residual, no LN | Residual helps gradient flow |
| vq_ssr | VQ-SSR (hybrid) | Discrete codebook + continuous grad |
| discrete | Gumbel-Softmax K=8 | Current best baseline |
| continuous | Linear projection | Structure matters vs raw continuous |
| no_comm | Zero messages | Lower bound |

### Phase 2: Scaling (50K episodes, best methods)
- Run top 3 methods for full 50K episodes
- Test message dim sweep: d=4, 8, 16

### Phase 3: Heterogeneous Models
- SmolLM2 + Qwen2.5-0.5B pair
- Tests whether SSR/VQ-SSR bridges different model families

## Evaluation Metrics

1. **Primary**: Mean episode reward across seeds (higher = better, negative distance)
2. **Sample efficiency**: Reward vs training episodes curve
3. **Message analysis**: Per-dimension variance, message norm, codebook utilization
4. **Robustness**: Std across seeds
5. **Statistical significance**: Report mean +/- std across 3 seeds

## Commands

```bash
# Phase 1: Full ablation (3 seeds x 7 methods x 10K episodes)
cd ~/auto-research/multi-agent-train
PYTHONPATH=. python scripts/compare_v2.py --episodes 10000 --seeds 42 123 456

# Phase 1 quick test (1 seed, fewer episodes)
PYTHONPATH=. python scripts/compare_v2.py --episodes 5000 --seeds 42

# Single method test
PYTHONPATH=. python scripts/compare_v2.py --episodes 5000 --seeds 42 --methods vq_ssr discrete
```
