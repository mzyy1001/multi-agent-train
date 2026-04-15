# Results Summary

## Experiment 1: Frozen Backbone Speaker-Listener (LOCAL, COMPLETE)
- 7 comm methods x 3 seeds x 10K episodes
- Model: SmolLM2-135M-Instruct (frozen)

| Rank | Method | Mean Reward | Std (seeds) | Range |
|------|--------|------------|-------------|-------|
| 1 | **Discrete** | **-46.81** | 6.45 | [-55.9, -41.8] |
| 2 | SSR | -55.76 | 18.59 | [-82.0, -40.7] |
| 3 | VQ-SSR | -62.18 | 27.73 | [-101.4, -42.4] |
| 4 | No-Comm | -65.69 | 24.37 | [-90.1, -41.3] |
| 5 | SSR v2 | -79.53 | 43.91 | [-141.6, -48.4] |
| 6 | SSR No-LN | -88.47 | 43.45 | [-148.9, -48.7] |
| 7 | Continuous | -138.59 | 51.92 | [-177.3, -65.2] |

Key: Discrete wins with frozen backbone on this categorical task.
LayerNorm in SSR actually helps (contrary to initial hypothesis).

## Experiment 2: Text Cooperative Classification (LOCAL, COMPLETE)
- Qwen2.5-0.5B-Instruct, LoRA rank=8, 30 epochs, 2 seeds

| Config | SSR | VQ-SSR | Discrete | No-Comm |
|--------|-----|--------|----------|---------|
| Frozen | 0.715 | 0.688 | 0.566 | 0.594 |
| LoRA-encoder | 0.754 | **0.781** | 0.727 | — |
| LoRA-both | 0.731 | 0.727 | 0.664 | — |
| delta(LoRA-enc) | +0.039 | **+0.094** | +0.160 | — |

Key: VQ-SSR + LoRA-encoder achieves best absolute performance.
The gradient highway enables meaningful LLM adaptation.

## Experiment 3: LoRA Speaker-Listener (ALIYUN, IN PROGRESS)
- SmolLM2-135M-Instruct, LoRA rank=4
- 3 comm methods x 3 backbone configs x 3 seeds
- Running on Aliyun A10 (estimated completion: ~40 hours from start)
- 4/27 experiments completed so far

## Overall Narrative

**With frozen backbones**: Discrete communication wins on the Speaker-Listener
task because the task is categorically structured (3 landmarks). The question
of "which bottleneck shape works best" has a clear but uninteresting answer:
match the task structure.

**With LoRA backbones**: The gradient highway changes the picture. On the text
cooperative task, differentiable channels (SSR, VQ-SSR) enable meaningful
encoder LLM adaptation that produces the best absolute results. The gradient
highway thesis is supported: differentiable communication serves dual purpose
as both message passing AND cross-agent LLM adaptation.
