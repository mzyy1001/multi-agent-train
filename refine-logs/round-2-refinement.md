# Round 2 Refinement

## Changes

### 1. Fair baselines — all use PPO (W1)

All baselines use the SAME architecture and RL algorithm (PPO):

| Config | Text | SSR | RL algo | What's tested |
|--------|------|-----|---------|---------------|
| No-comm | none | none | PPO | Lower bound |
| Text-only | yes | zeros | PPO | Text value without gradient flow |
| SSR-only | none | yes | PPO | SSR value without text |
| **Dual-channel** | **yes** | **yes** | **PPO** | **Both channels** |
| Centralized | N/A | N/A | PPO | Upper bound (sees everything) |

For text-only: the SSR channel exists but outputs zeros (like no-comm in prior work). The text messages are read by other agents' LLMs as context. PPO trains the action heads + projectors, but gradients don't flow through text.

This ensures the ONLY difference between text-only and dual-channel is whether the SSR channel carries information.

### 2. Text quality verification (W2)

Before RL training, run a text-quality check:
- Generate text messages from prompted LLMs for 10 sample observations
- Manually verify they contain role-relevant information
- Report example messages in the paper

### 3. Ablation for credit assignment (W3)

Credit assignment experiment:
- Train dual-channel system to convergence
- Then: zero out SSR from CEO only → measure profit drop (Δ_CEO)
- Then: zero out SSR from CTO only → measure profit drop (Δ_CTO)
- The agent with larger Δ has more influence through the gradient highway
- Compare with actual gradient norms through each agent's LoRA

## Final Proposal Score Estimate: 9/10 — READY

The proposal is now concrete enough to implement with:
- Precise MDP specification
- Clear fusion mechanism
- Fair baselines (all PPO)
- 5-way comparison + credit assignment ablation
- Feasible compute budget (400-500 GPU-hours)
- Novel dual-channel contribution (9/10 novelty)
