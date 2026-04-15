# Round 1 Review (Self-Review as Senior ML Reviewer)

## Scores

| Dimension | Score | Notes |
|-----------|-------|-------|
| Problem Fidelity | 9 | Clear bottleneck: text blocks gradients in multi-agent LLM RL |
| Method Specificity | 6 | System overview is clear but training details are vague |
| Contribution Quality | 8 | Dual-channel is genuinely novel and focused |
| Frontier Leverage | 8 | Good use of LoRA + SSR from prior work |
| Feasibility | 5 | Business simulation is underspecified — what exactly are the state/action spaces? |
| Validation Focus | 7 | Three claims with clear experiments |
| Venue Readiness | 7 | Novel but needs concrete method details |

**Overall: 7.1/10**
**Verdict: REVISE**

## Critical Issues

### W1 (CRITICAL): Environment is underspecified
The "company simulation" is hand-wavy. What are the exact state variables? What are the action spaces? How is profit computed? Without a concrete environment spec, the proposal is a concept, not a method.

**Fix**: Define a precise MDP: state space (what each agent observes), action space (discrete/continuous choices per role), transition dynamics (how decisions affect next quarter), reward function (exact profit formula).

### W2 (IMPORTANT): How do text and SSR interact at the receiver?
The proposal says the receiver gets "both" but doesn't specify the fusion mechanism. Does the receiver concatenate text embeddings with SSR vectors? Does SSR modulate attention over text? This is the core architectural question.

**Fix**: Define the exact fusion mechanism. Simplest: receiver's LLM processes text as context, then the SSR vector is fused with the LLM's output through an adapter (like our prior work).

### W3 (IMPORTANT): Text channel training is unclear
"Frozen or imitation learning" is too vague. If text is frozen (from prompted LLM), the text quality depends on prompt engineering. If trained by imitation, where does the imitation data come from?

**Fix**: Start with prompted LLM (frozen text). This is simpler and isolates the SSR contribution. Text improvement can be future work.

### W4 (MODERATE): Scale concern — 3 agents × 0.5B each = 1.5B parameters
Training 3 LoRA-adapted LLMs simultaneously with PPO requires significant GPU memory. Each forward pass goes through 3 models.

**Fix**: Use SmolLM-135M for initial experiments (3 × 135M = 405M, fits on one GPU). Scale to Qwen-0.5B for final results.

### W5 (MODERATE): Missing "SSR-only" baseline
The proposal compares dual-channel vs text-only but doesn't include SSR-only (drop text entirely). This is needed to show text actually contributes.

**Fix**: Add SSR-only baseline. If SSR-only matches dual-channel, text is redundant (bad for the thesis).

## Simplification Opportunities
1. Start with 2 agents instead of 3 to reduce complexity
2. Use a simpler environment (e.g., matrix game with information asymmetry) before the company simulation
3. Drop Stage 1 (text pretraining) — just use prompted LLM

## Drift Warning: NONE
The proposal stays focused on the core problem.
