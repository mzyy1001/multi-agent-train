# Novelty Check Report

## Proposed Method
Using a differentiable inter-agent communication channel (SSR/VQ-SSR) as a "gradient highway"
to enable LoRA fine-tuning of LLM backbones in cooperative MARL, where one agent's policy loss
provides a cross-agent training signal that shapes another agent's LLM representations.

## Core Claims
1. **Differentiable comm as gradient highway for LLM adaptation** — Novelty: HIGH — Closest: C2C (ICLR 2026)
2. **Cross-agent LoRA training through communication loss** — Novelty: HIGH — Closest: MARFT (2025)
3. **LoRA fine-tuning in cooperative MARL with continuous messages** — Novelty: HIGH — Closest: CORY (NeurIPS 2024)
4. **Frozen vs LoRA-speaker vs LoRA-both comparison framework** — Novelty: MEDIUM
5. **VQ-SSR bridging discrete + continuous for LLM adaptation** — Novelty: HIGH

## Closest Prior Work
| Paper | Year | Venue | Overlap | Key Difference |
|-------|------|-------|---------|----------------|
| C2C | 2026 | ICLR | Differentiable LLM-to-LLM comm | NLP tasks, KV-cache level, no RL |
| MARFT | 2025 | arXiv | LoRA per agent in multi-agent LLM | Text communication (non-differentiable) |
| CORY | 2024 | NeurIPS | Cooperative multi-agent LLM fine-tuning | Sequential text interaction |
| DIAL | 2016 | NeurIPS | Differentiable inter-agent comm | Small networks, no LLMs |
| VQ-VIB | 2022 | RSS-W | VQ for emergent communication | No LLMs, no backbone adaptation |

## Overall Assessment
- Score: 8/10
- Recommendation: PROCEED
- Key differentiator: The integration of differentiable communication + LoRA + cross-agent gradient flow in MARL is novel
- Risk: C2C closest competitor, but operates at fundamentally different level (KV-cache for NLP vs latent messages for MARL)
