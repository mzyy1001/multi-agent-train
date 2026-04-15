# Research Proposal: Dual-Channel Communication for Multi-Agent LLM Cooperative RL

## Problem Anchor
- **Bottom-line problem**: Multi-agent LLM systems communicate through discrete text, blocking gradient flow between agents. This forces reliance on high-variance REINFORCE estimators for RL-based multi-agent optimization, making credit assignment across agents intractable for complex cooperative tasks.
- **Must-solve bottleneck**: There is no mechanism for end-to-end differentiable training across multiple LLM agents that also preserves interpretable text communication. Existing work either uses text-only (MARFT, CORY — no gradient flow) or replaces text entirely (C2C — loses interpretability).
- **Non-goals**: We do not aim to build a production-ready company simulator. We do not aim to replace existing multi-agent frameworks (AutoGen, CrewAI). We do not aim to train agents from scratch — we use LoRA on pretrained LLMs.
- **Constraints**: Compute: 1-8 GPUs (A100 or RTX 4090). Models: 0.5B-3B parameters. Timeline: 4-6 weeks. No access to proprietary APIs for training (open-source models only).
- **Success condition**: Show that dual-channel (text+SSR) agents outperform text-only and SSR-only agents on a multi-agent cooperative task with long-horizon reward, and that the SSR channel enables meaningful credit assignment across agents.

## Technical Gap

Current multi-agent LLM systems face a fundamental trilemma:
1. **Text-only** (MARFT, CORY, AutoGen): Interpretable but no gradient flow → REINFORCE with O(n) variance, poor credit assignment
2. **Differentiable-only** (C2C, our prior SSR work): Gradient flow but loses interpretability → can't inspect agent reasoning
3. **Classical MARL** (CommNet, DIAL, TarMAC): Gradient flow with small networks but doesn't leverage LLM representations

No existing work combines interpretable text communication with a parallel differentiable channel for RL optimization in LLM multi-agent systems.

The gap: **a dual-channel architecture that uses text for interpretability and a parallel differentiable SSR channel for end-to-end RL gradient flow across LLM agents**.

## Method Thesis
- **One-sentence thesis**: A parallel differentiable SSR channel alongside natural language enables end-to-end RL optimization of multi-agent LLM systems while preserving interpretable text communication.
- **Why smallest adequate intervention**: We add only one component (SSR channel) to existing text-based multi-agent LLM frameworks. Everything else (LLM backbones, text generation, RL algorithm) is reused.
- **Why timely**: Multi-agent LLM systems are exploding (AutoGen, CrewAI, MARTI) but all hit the text-only wall for RL training. Our prior work proved SSR+LoRA works for gradient flow on novel domains.

## Contribution Focus
- **Dominant contribution**: Dual-channel (text+SSR) architecture enabling end-to-end RL across LLM agents
- **Supporting contribution**: Empirical evidence that the SSR gradient highway solves credit assignment in multi-agent cooperative tasks
- **Explicit non-contributions**: Not a new RL algorithm. Not a new LLM architecture. Not a production system.

## Proposed Method

### Complexity Budget
- **Frozen/reused**: LLM backbones (Qwen-0.5B or SmolLM-135M), PPO algorithm, text generation pipeline
- **New trainable**: LoRA adapters (per agent), SSR channel encoder/decoder, role-specific action heads
- **Intentionally excluded**: No attention over other agents' hidden states (too expensive), no graph neural network over agent topology (unnecessary)

### System Overview

```
Each Agent i:
  Input: observation_i (role-specific slice of company report)
  
  Text path (discrete, interpretable):
    obs_i → LLM_i → text_message_i (sampled tokens, no gradient)
    
  SSR path (differentiable):
    obs_i → LLM_i(LoRA) → Projector_i → SSR_encoder → ssr_message_i (d-dim continuous)
    
  Reception:
    text_messages_from_others → LLM_i reads them as context
    ssr_messages_from_others → SSR_decoder → fused_state
    
  Decision:
    fused_state → ActionHead_i → action_i (role-specific decisions)
    
  Training:
    reward (company profit at end of episode) → PPO loss
    → backprop through SSR path into all agents' LoRA weights
    text path: NOT trained by RL (frozen or imitation learning)
```

### Core Mechanism: Dual-Channel Communication

Each communication round between agents has TWO parallel channels:

1. **Text channel**: Agent generates natural language message via standard LLM decoding. This is discrete (sampled tokens), interpretable, and NOT part of the RL gradient path. It can be trained separately via supervised imitation or left frozen.

2. **SSR channel**: Agent's LLM hidden state (from the same observation) is projected through an SSR encoder into a d-dimensional continuous vector. This vector is passed to receiving agents through an SSR decoder that fuses it with the receiver's state. The entire SSR path is differentiable.

The receiving agent gets BOTH: it reads the text message as LLM context AND receives the SSR vector through the differentiable adapter. The final decision is based on the fused representation.

**Why dual**: Text provides interpretability and leverages the LLM's language capabilities. SSR provides the gradient highway for RL training. Neither alone is sufficient — text can't carry gradients, SSR can't be inspected by humans.

### Training Plan

**Stage 1: Text pretraining** (optional)
- Train agents to generate sensible role-specific messages via supervised imitation on synthetic company dialogues
- Or start from a prompted LLM that already generates reasonable text

**Stage 2: RL with SSR gradient highway**
- Freeze text generation (or keep it very lightly updated)
- Train LoRA adapters + SSR encoder/decoder + action heads via PPO
- Company-wide reward (multi-quarter profit) provides the RL signal
- Gradients flow through SSR channel into all agents' LoRA weights
- PPO with GAE for long-horizon credit assignment

**Stage 3: Analysis**
- Compare dual-channel vs text-only vs SSR-only
- Measure credit assignment quality (per-agent gradient norms)
- Analyze what SSR vectors encode vs what text messages say

### Environment: Multi-Quarter Business Simulation

**Simplified but meaningful**:
- 3 agents: CEO (strategy), CTO (product), Sales (revenue)
- Each quarter: agents observe their role-specific report slice, communicate (2 rounds), make decisions
- Decisions: CEO sets priorities (invest in R&D / marketing / cost-cutting), CTO allocates engineering effort, Sales sets pricing
- Environment simulates: revenue = f(product_quality, price, market_conditions), costs = f(engineering_effort, marketing_spend), profit = revenue - costs
- Reward: cumulative profit over 8-12 quarters (long-horizon)
- Information asymmetry: CEO sees market trends, CTO sees tech metrics, Sales sees customer data

### Failure Modes
- SSR channel might dominate, making text irrelevant → monitor text influence via ablation
- LoRA overfitting on small simulation → use SSR bottleneck regularization (proven in prior work)
- Environment too simple → can increase complexity (add competitors, random events)

### Novelty Argument
- **vs MARFT**: MARFT uses text-only communication → no gradient flow between agents
- **vs C2C**: C2C replaces text with differentiable KV-cache fusion → loses interpretability
- **vs MARTI**: MARTI focuses on code generation with tree search → not cooperative RL
- **vs CommNet/DIAL**: Classical MARL with small networks → doesn't leverage LLM representations
- **vs our prior work**: SSR on simple retrieval tasks → this scales to multi-agent cooperative RL with long-horizon reward

## Claim-Driven Validation

### Claim 1: Dual-channel outperforms text-only on long-horizon cooperative tasks
- **Experiment**: Compare dual-channel vs text-only agents on 12-quarter business simulation
- **Baseline**: Text-only agents with REINFORCE for RL
- **Metric**: Cumulative profit, sample efficiency
- **Expected**: Dual-channel converges faster and higher due to low-variance gradients through SSR

### Claim 2: SSR gradient highway enables credit assignment
- **Experiment**: Measure per-agent LoRA gradient norms → which agent's backbone adapts most?
- **Ablation**: Remove SSR from one agent → how does company performance degrade?
- **Expected**: Removing SSR from CEO hurts most (strategic decisions have highest leverage)

### Claim 3: Text and SSR channels carry complementary information
- **Analysis**: Compare what text messages say vs what SSR vectors encode
- **Method**: Probe SSR vectors with linear classifiers, compare with text content analysis
- **Expected**: SSR encodes quantitative/relational info that text messages omit or distort

## Compute & Timeline Estimate
- **GPU-hours**: ~200-500 hours (3 agents × LoRA × PPO × multiple seeds)
- **Data**: Synthetic business simulation (no external data needed)
- **Timeline**: 4-6 weeks (1 week env, 2 weeks training, 1-2 weeks analysis + paper)

---
# REFINEMENTS (incorporated from review rounds)

## Concrete Environment MDP

**State space**: CEO (9-dim: market, revenue, costs), CTO (7-dim: product quality, tech debt, bugs), Sales (7-dim: satisfaction, churn, market share)

**Action space**: CEO (budget allocation 3-dim), CTO (engineering focus 3-dim), Sales (pricing strategy 3-dim). All continuous, trained by PPO.

**Reward**: Cumulative discounted profit over 8 quarters (γ=0.95)

## Fusion Mechanism
```
h_fused = Adapter(concat(h_text_from_LLM, SSR_decoded))  → ActionHead → action
```

## Fair Baselines (all use PPO)

| Config | Text | SSR | What's tested |
|--------|------|-----|---------------|
| No-comm | none | none | Lower bound |
| Text-only | yes | zeros | Text value without gradients |
| SSR-only | none | yes | SSR value without text |
| Dual-channel | yes | yes | Both (our method) |
| Centralized | N/A | N/A | Upper bound |

## Credit Assignment Ablation
- Zero out SSR per agent → measure profit drop
- Compare gradient norms across agents' LoRA weights

## Implementation Plan
1. Start with 2 agents (CEO + CTO) on SmolLM-135M
2. Scale to 3 agents + Qwen-0.5B after validation
3. ~400-500 GPU-hours total

## Novelty: 9/10
No prior work combines text + differentiable channels in multi-agent LLM RL.
