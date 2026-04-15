# Round 1 Refinement

## Problem Anchor (unchanged)
- Bottom-line problem: Text-only multi-agent LLM systems can't do end-to-end RL optimization
- Must-solve bottleneck: No gradient flow between agents through discrete text
- Success condition: Dual-channel outperforms text-only and SSR-only on cooperative task

## Anchor Check
- Original bottleneck: preserved (text blocks gradients)
- Reviewer suggestions: all valid, no drift

## Changes Made

### 1. Concrete Environment Specification (W1)

**Business Quarterly Simulation MDP**:

**State space (per agent, per quarter)**:
- CEO observes: market_trend (3-dim), competitor_action (3-dim), company_revenue (1), company_costs (1), quarter_number (1) = 9 dims
- CTO observes: product_quality (1), tech_debt (1), team_size (1), bug_count (1), feature_backlog (3-dim) = 7 dims  
- Sales observes: customer_satisfaction (1), churn_rate (1), market_share (1), pipeline_value (1), price_sensitivity (3-dim) = 7 dims

Each observation is converted to a text prompt for the LLM (like Speaker-Listener).

**Action space (per agent, per quarter)**:
- CEO: allocate_budget = [R&D%, Marketing%, Operations%] (3-dim continuous, must sum to 1)
- CTO: engineering_focus = [new_features, bug_fixes, infrastructure] (3-dim continuous, must sum to 1)
- Sales: pricing_strategy = [price_level, discount_rate, target_segment] (3-dim continuous)

**Transition dynamics**:
```
product_quality += CTO_new_features * R&D_budget - tech_debt_decay
tech_debt += CTO_new_features * 0.3 - CTO_bug_fixes * 0.5
customer_satisfaction = f(product_quality, price_level, support_quality)
churn_rate = sigmoid(-customer_satisfaction + price_sensitivity)
revenue = market_size * market_share * (1 - churn_rate) * price_level
costs = team_size * salary + marketing_budget + operations_budget
profit = revenue - costs
market_share += marketing_effect - competitor_effect + product_quality_effect
```

**Reward**: cumulative discounted profit over 8 quarters (γ=0.95)

### 2. Fusion Mechanism (W2)

Receiver gets text + SSR through two paths:
```
Text path:  text_messages → LLM context window → h_text (LLM output)
SSR path:   ssr_vectors → SSR_decoder → s_ssr (decoded SSR)

Fusion:     h_fused = Adapter(concat(h_text, s_ssr))  → ActionHead → action
```

The Adapter is a 2-layer MLP that fuses the text-derived LLM hidden state with the decoded SSR vector. This is identical to our prior ReceiverAdapter but with text context added to the LLM input.

### 3. Text Channel: Prompted LLM, Frozen (W3)

Text generation uses a fixed prompt template per role:
- CEO: "You are the CEO. Market trend: {obs}. What strategic direction should the company take? Respond in 1-2 sentences."
- CTO: "You are the CTO. Product metrics: {obs}. What should the engineering team focus on?"
- Sales: "You are the Sales lead. Customer data: {obs}. What pricing and sales strategy do you recommend?"

Text is generated via greedy decoding (deterministic), NOT trained by RL. This isolates the SSR contribution.

### 4. Start with 2 agents (W4 + simplification)

Initial experiments: CEO + CTO only (2 agents, simpler dynamics)
- CEO: sees market + revenue → allocates budget
- CTO: sees tech metrics → allocates engineering effort
- Communication: CEO tells CTO priorities, CTO tells CEO technical status

Scale to 3 agents (add Sales) after 2-agent results validate the approach.

### 5. Add SSR-only baseline (W5)

Full comparison:
1. **Text-only**: agents communicate via text, RL with REINFORCE (no gradient through text)
2. **SSR-only**: agents communicate via SSR vectors only, RL with PPO (gradient through SSR)
3. **Dual-channel**: text + SSR, RL with PPO through SSR (text frozen)
4. **No communication**: agents act independently
5. **Centralized**: single agent sees all observations (upper bound)

## Revised Compute Estimate
- 2 agents × SmolLM-135M: ~270M params, fits on 1 GPU easily
- PPO training: ~100 GPU-hours for initial results
- Scale to Qwen-0.5B + 3 agents: ~300 GPU-hours
- Total: ~400-500 GPU-hours
