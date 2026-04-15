# Idea: Dual-Channel Multi-Agent Company with SSR Gradient Highway

## Core Concept

Build a multi-agent LLM system structured as a company where agents (CEO, CTO, COO, etc.)
cooperate to maximize company profit. Two communication channels:

1. **Text channel** (forward only): Natural language communication between agents.
   Human-readable, interpretable, but discrete — no gradient flow.

2. **SSR channel** (differentiable): Parallel continuous vector channel.
   Carries gradient signal for RL training. Enables end-to-end optimization
   across all agents using a single company-wide reward.

## Why This Works

- **Credit assignment**: Company profit is a single scalar shared by all agents.
  Without SSR, each agent must learn independently (REINFORCE, high variance).
  With SSR, gradients flow from the reward through all agents' decisions.

- **Dual purpose**: Text for interpretability + SSR for training.
  At deployment, you can even remove the SSR channel — the agents have already
  learned good policies through the gradient highway during training.

- **Novelty**: No prior work combines text + differentiable channels in multi-agent LLMs.
  C2C replaces text; MARFT keeps text only. We use BOTH.

## Company Structure

```
         CEO (strategic decisions)
        /    \
      CTO    COO (operations)
      |       |
    Dev1    Sales1
    Dev2    Sales2
```

Each agent:
- Has its own LLM backbone (with LoRA)
- Reads text messages from other agents
- Sends text messages (discrete, for interpretability)
- ALSO sends SSR vectors through differentiable channel
- The SSR vectors carry compressed "intent" that the RL optimizer can backprop through

## Rich Structured Reward (not a simple scalar)

The "reward" is NOT a single profit number — it's a rich multi-dimensional report:
- Revenue breakdown (by product, by region)
- Cost structure (engineering, operations, marketing)
- Customer metrics (satisfaction, churn, NPS)
- Technical metrics (uptime, bug count, latency)
- Market data (competitor moves, market share trends)

Each agent sees a DIFFERENT SLICE of this report:
- CEO: high-level P&L + market trends
- CTO: technical metrics + engineering costs
- COO: operational data + headcount + delivery timelines  
- Sales: revenue details + customer feedback

This creates GENUINE information asymmetry:
- CEO can't see bug count → needs CTO to communicate
- CTO can't see customer churn → needs Sales to communicate
- Text channel: biased/lossy human summaries ("things are fine")
- SSR channel: unfiltered gradient signal that bypasses reporting bias

The RL reward is computed from the FULL report, but each agent only sees
their slice. The gradient flows through SSR to help agents learn WHAT to
communicate and HOW to interpret partial information.

## Training Loop

1. Company receives a scenario + market conditions
2. Each agent reads their slice of the company report
3. Multi-round communication: text (interpretable) + SSR (differentiable)
4. Each agent makes role-specific decisions
5. Environment simulates outcomes → generates next period's report
6. RL loss (from full report) backpropagates through SSR into all agents' LoRA
7. Text channel trained by imitation learning or left frozen

## Key Research Questions

1. Does the SSR gradient highway improve company performance vs text-only?
2. Does LoRA adaptation through SSR enable specialization (CEO becomes strategic, CTO becomes technical)?
3. Can SSR capture "unspoken" coordination that text misses?
4. Does the dual-channel approach scale with number of agents?

## Implementation Plan

- Start with 2-3 agents on a simple business simulation
- Use a text-based business game or create one
- Each agent: Qwen-0.5B with LoRA + SSR channel
- Training: PPO with shared reward, gradients through SSR only
- Evaluate: company performance with/without SSR channel

## Comparison to Current Work

Current paper: SSR on simple retrieval/NLI tasks — low complexity
This idea: SSR on multi-agent company simulation — high complexity, real motivation

The dual-channel framing is much more publishable because:
1. It explains WHY you need differentiable communication (gradient highway for RL)
2. It explains WHY you also need text (interpretability, human oversight)
3. The company setting is intuitive and practically relevant
