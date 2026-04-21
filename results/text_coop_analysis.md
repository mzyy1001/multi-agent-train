# Text Cooperative Task — Gradient Highway Analysis

## Setup
- Model: Qwen2.5-0.5B-Instruct
- Task: Split-context binary entailment classification
- Encoder sees context, Classifier sees hypothesis, communicate via message
- 2 seeds (42, 123), 30 epochs, lr=1e-4, LoRA rank=8

## Results Summary (Val Accuracy, averaged over 2 seeds)

| Config | SSR | VQ-SSR | Discrete | No-Comm |
|--------|-----|--------|----------|---------|
| Frozen | 0.7148 | 0.6875 | 0.5664 | 0.5938 |
| LoRA-encoder | 0.7539 | **0.7812** | 0.7266 | — |
| LoRA-both | 0.7305 | 0.7266 | 0.6641 | — |
| delta(LoRA-enc) | +0.039 | **+0.094** | +0.160 | — |

## Key Findings

1. **VQ-SSR + LoRA-encoder achieves best absolute performance (0.7812)**
   - The gradient highway enables meaningful encoder LLM adaptation
   - VQ-SSR's discrete codebook helps compress text-relevant info effectively

2. **SSR frozen already good (0.7148)** — the differentiable channel
   carries text semantics well even without backbone adaptation

3. **Discrete frozen is poor (0.5664)** — K=8 one-hot symbols can't carry
   rich text semantics. But LoRA helps a lot (+16%) by making the encoder
   produce features that compress better into discrete symbols.

4. **LoRA-both sometimes hurts** — adding LoRA to both agents overfits on
   this relatively small dataset (1296 train samples). LoRA-encoder-only
   is the sweet spot.

5. **Gradient highway thesis partially supported**: LoRA-encoder benefits
   ALL comm types, but the absolute best result comes from differentiable
   channels (VQ-SSR > SSR > discrete). The gradient highway provides
   the best overall result.

## Gradient Highway Interpretation
- With frozen backbone, SSR's gradient highway from classifier to encoder
  is limited to projector/comm weights only
- With LoRA-encoder, the highway extends into the encoder's LLM backbone
- The best combination (VQ-SSR + LoRA) benefits from both the discrete
  codebook structure AND full gradient flow to the encoder LLM
