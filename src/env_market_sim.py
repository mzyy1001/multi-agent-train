"""Market Simulation Environment for Multi-Agent Company RL.

A multi-quarter business simulation where agents (CEO, CTO, Sales) make
decisions each quarter and the market responds with realistic dynamics.

Key features:
- Information asymmetry: each agent sees only their role-specific metrics
- Long-horizon reward: cumulative profit over 8-12 quarters
- Market dynamics: competitors, demand curves, tech debt, customer churn
- Stochastic: random market events, competitor moves, demand noise

The environment is designed so that:
1. Communication is NECESSARY (no single agent sees everything)
2. Long-term planning matters (short-term greed hurts long-term profit)
3. The reward is rich (quarterly financial report, not just a scalar)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketState:
    """Full market state (not all visible to any single agent)."""
    quarter: int = 0

    # Product
    product_quality: float = 0.5        # 0-1, how good the product is
    tech_debt: float = 0.1              # 0-1, accumulated technical debt
    feature_count: int = 3              # number of features shipped
    bug_count: int = 2                  # outstanding bugs

    # Market
    market_size: float = 1000.0         # total addressable market ($)
    market_share: float = 0.1           # 0-1, our share
    competitor_quality: float = 0.5     # 0-1, competitor product quality
    market_growth: float = 0.05         # quarterly market growth rate
    demand_elasticity: float = 1.5      # price sensitivity

    # Customers
    customer_count: int = 100
    customer_satisfaction: float = 0.6  # 0-1
    churn_rate: float = 0.05            # quarterly churn
    nps_score: float = 30.0             # -100 to 100

    # Financial
    revenue: float = 100.0
    costs: float = 80.0
    profit: float = 20.0
    cash: float = 200.0
    price: float = 10.0

    # Team
    engineering_team: int = 5
    sales_team: int = 3
    salary_per_person: float = 15.0     # per quarter

    # History
    quarterly_profits: list = field(default_factory=list)


class MarketSimEnv:
    """Multi-quarter business simulation.

    Each quarter:
    1. Agents observe their role-specific metrics
    2. Agents communicate (text + SSR)
    3. Agents make decisions (budget allocation, engineering focus, pricing)
    4. Environment simulates market response
    5. New quarterly report generated

    Returns per-agent observations and company-wide reward.
    """

    def __init__(
        self,
        n_quarters: int = 8,
        seed: int = 42,
        difficulty: str = "medium",
    ):
        self.n_quarters = n_quarters
        self.rng = np.random.RandomState(seed)
        self.difficulty = difficulty
        self.state: Optional[MarketState] = None

        # Difficulty settings
        self._competitor_aggressiveness = {
            "easy": 0.02, "medium": 0.05, "hard": 0.1
        }[difficulty]
        self._market_volatility = {
            "easy": 0.02, "medium": 0.05, "hard": 0.1
        }[difficulty]

    def reset(self, seed: int = None) -> dict[str, np.ndarray]:
        """Reset to quarter 0. Returns initial observations per agent."""
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        self.state = MarketState(
            quarter=0,
            product_quality=0.4 + self.rng.uniform(-0.1, 0.1),
            tech_debt=0.1 + self.rng.uniform(0, 0.1),
            market_size=1000 + self.rng.uniform(-100, 100),
            market_share=0.08 + self.rng.uniform(-0.02, 0.02),
            competitor_quality=0.5 + self.rng.uniform(-0.1, 0.1),
            customer_satisfaction=0.5 + self.rng.uniform(-0.1, 0.1),
            cash=200 + self.rng.uniform(-50, 50),
            price=10 + self.rng.uniform(-2, 2),
        )
        self.state.customer_count = int(self.state.market_size * self.state.market_share)
        self.state.revenue = self.state.customer_count * self.state.price
        self.state.costs = (self.state.engineering_team + self.state.sales_team) * self.state.salary_per_person
        self.state.profit = self.state.revenue - self.state.costs
        self.state.quarterly_profits = []

        return self._get_observations()

    def step(
        self,
        ceo_action: np.ndarray,    # [rd_pct, marketing_pct, ops_pct] must sum to 1
        cto_action: np.ndarray,    # [new_features, bug_fixes, infra] must sum to 1
        sales_action: np.ndarray,  # [price_delta, discount_rate, outreach_effort]
    ) -> tuple[dict[str, np.ndarray], float, bool, dict]:
        """
        Execute one quarter.

        Returns: (observations, reward, done, info)
        """
        s = self.state
        s.quarter += 1

        # === Normalize actions ===
        ceo_action = np.clip(ceo_action, 0.01, 1.0)
        ceo_action = ceo_action / ceo_action.sum()  # ensure sums to 1
        rd_pct, marketing_pct, ops_pct = ceo_action

        cto_action = np.clip(cto_action, 0.01, 1.0)
        cto_action = cto_action / cto_action.sum()
        feat_focus, bugfix_focus, infra_focus = cto_action

        price_delta = np.clip(sales_action[0], -0.3, 0.3)
        discount_rate = np.clip(sales_action[1], 0.0, 0.5)
        outreach_effort = np.clip(sales_action[2], 0.0, 1.0)

        # === Budget allocation ===
        total_budget = s.revenue * 0.8  # 80% of revenue reinvested
        rd_budget = total_budget * rd_pct
        marketing_budget = total_budget * marketing_pct
        ops_budget = total_budget * ops_pct

        # === Engineering effects ===
        # New features improve product quality
        feature_improvement = feat_focus * rd_budget / 100.0 * (1 - s.tech_debt)
        s.product_quality = np.clip(s.product_quality + feature_improvement * 0.1, 0, 1)
        s.feature_count += max(0, int(feature_improvement * 5))

        # Bug fixes reduce bugs and tech debt
        bugs_fixed = int(bugfix_focus * s.engineering_team * 2)
        s.bug_count = max(0, s.bug_count - bugs_fixed)
        s.tech_debt = np.clip(s.tech_debt - bugfix_focus * 0.05, 0, 1)

        # Infrastructure reduces future tech debt accumulation
        infra_effect = infra_focus * 0.03

        # New features ADD tech debt
        s.tech_debt = np.clip(s.tech_debt + feat_focus * 0.03 - infra_effect, 0, 1)

        # Bugs accumulate from tech debt
        new_bugs = self.rng.poisson(s.tech_debt * 3)
        s.bug_count += new_bugs

        # === Pricing effects ===
        s.price = np.clip(s.price * (1 + price_delta), 1.0, 50.0)
        effective_price = s.price * (1 - discount_rate)

        # === Customer dynamics ===
        # Satisfaction depends on product quality, bugs, and price fairness
        quality_factor = s.product_quality * (1 - s.bug_count / 20.0)
        price_fairness = 1.0 - abs(effective_price - 10.0) / 20.0
        s.customer_satisfaction = np.clip(
            0.3 * quality_factor + 0.3 * price_fairness + 0.4 * s.customer_satisfaction,
            0, 1
        )

        # Churn depends on satisfaction
        s.churn_rate = np.clip(0.15 - s.customer_satisfaction * 0.12, 0.01, 0.3)
        churned = int(s.customer_count * s.churn_rate)

        # New customers from marketing and word-of-mouth
        marketing_reach = marketing_budget / 50.0 * outreach_effort
        organic_growth = s.customer_satisfaction * s.nps_score / 100.0
        new_customers = int(marketing_reach * 5 + organic_growth * s.customer_count * 0.02)
        new_customers = max(0, new_customers + self.rng.randint(-5, 10))

        s.customer_count = max(1, s.customer_count - churned + new_customers)

        # NPS evolves
        s.nps_score = np.clip(
            s.nps_score + (s.customer_satisfaction - 0.5) * 10 + self.rng.normal(0, 3),
            -100, 100
        )

        # === Market dynamics ===
        s.market_size *= (1 + s.market_growth + self.rng.normal(0, self._market_volatility))
        s.market_share = np.clip(s.customer_count / max(1, s.market_size / s.price), 0, 1)

        # Competitor adapts
        s.competitor_quality = np.clip(
            s.competitor_quality + self.rng.normal(self._competitor_aggressiveness, 0.02),
            0, 1
        )
        # If competitor is better, we lose share
        if s.competitor_quality > s.product_quality:
            s.market_share *= 0.97

        # === Random market events (stochastic) ===
        event_roll = self.rng.random()
        event_info = None
        if event_roll < 0.05:
            # Market crash
            s.market_size *= 0.85
            event_info = "market_crash"
        elif event_roll < 0.10:
            # New competitor enters
            s.competitor_quality += 0.1
            event_info = "new_competitor"
        elif event_roll < 0.15:
            # Viral growth
            s.customer_count = int(s.customer_count * 1.3)
            event_info = "viral_growth"
        elif event_roll < 0.18:
            # Key employee leaves
            s.engineering_team = max(2, s.engineering_team - 1)
            event_info = "employee_leaves"

        # === Financial calculation ===
        s.revenue = s.customer_count * effective_price
        personnel_cost = (s.engineering_team + s.sales_team) * s.salary_per_person
        s.costs = personnel_cost + ops_budget * 0.5  # ops budget is partially cost
        s.profit = s.revenue - s.costs
        s.cash += s.profit
        s.quarterly_profits.append(s.profit)

        # === Hiring (from ops budget) ===
        if ops_budget > s.salary_per_person * 2 and s.cash > s.salary_per_person * 4:
            if self.rng.random() < 0.3:
                s.engineering_team += 1
            if self.rng.random() < 0.2:
                s.sales_team += 1

        # === Done? ===
        done = s.quarter >= self.n_quarters or s.cash < 0

        # === Reward ===
        # Reward is the quarterly profit, normalized
        reward = s.profit / 100.0  # scale to reasonable range

        # Bankruptcy penalty
        if s.cash < 0:
            reward -= 10.0

        info = {
            "quarter": s.quarter,
            "profit": s.profit,
            "revenue": s.revenue,
            "costs": s.costs,
            "cash": s.cash,
            "market_share": s.market_share,
            "customer_count": s.customer_count,
            "product_quality": s.product_quality,
            "tech_debt": s.tech_debt,
            "customer_satisfaction": s.customer_satisfaction,
            "event": event_info,
            "cumulative_profit": sum(s.quarterly_profits),
        }

        return self._get_observations(), reward, done, info

    def _get_observations(self) -> dict[str, np.ndarray]:
        """Return role-specific observations for each agent."""
        s = self.state

        # CEO sees high-level strategy metrics
        ceo_obs = np.array([
            s.market_size / 2000.0,           # normalized market size
            s.market_share,                    # our market share
            s.market_growth,                   # growth rate
            s.competitor_quality,              # competitor strength
            s.revenue / 500.0,                 # normalized revenue
            s.costs / 500.0,                   # normalized costs
            s.profit / 200.0,                  # normalized profit
            s.cash / 500.0,                    # normalized cash
            s.quarter / self.n_quarters,       # progress through game
        ], dtype=np.float32)

        # CTO sees technical metrics
        cto_obs = np.array([
            s.product_quality,                 # product quality
            s.tech_debt,                       # technical debt
            s.feature_count / 20.0,            # normalized feature count
            s.bug_count / 20.0,                # normalized bug count
            s.engineering_team / 10.0,         # normalized team size
            s.quarter / self.n_quarters,       # progress
            s.revenue / 500.0,                 # revenue (limited financial view)
        ], dtype=np.float32)

        # Sales sees customer and revenue metrics
        sales_obs = np.array([
            s.customer_satisfaction,           # customer satisfaction
            s.churn_rate,                      # churn rate
            s.nps_score / 100.0,               # normalized NPS
            s.customer_count / 500.0,          # normalized customer count
            s.price / 20.0,                    # normalized price
            s.market_share,                    # market share
            s.quarter / self.n_quarters,       # progress
        ], dtype=np.float32)

        return {
            "ceo": ceo_obs,
            "cto": cto_obs,
            "sales": sales_obs,
        }

    def get_text_report(self) -> dict[str, str]:
        """Generate natural language reports for each agent."""
        s = self.state

        ceo_report = (
            f"Q{s.quarter} CEO Report: Market size ${s.market_size:.0f}, "
            f"our share {s.market_share*100:.1f}%, growth {s.market_growth*100:.1f}%. "
            f"Competitor quality {s.competitor_quality:.2f}. "
            f"Revenue ${s.revenue:.0f}, costs ${s.costs:.0f}, profit ${s.profit:.0f}. "
            f"Cash reserves ${s.cash:.0f}."
        )

        cto_report = (
            f"Q{s.quarter} CTO Report: Product quality {s.product_quality:.2f}, "
            f"tech debt {s.tech_debt:.2f}. "
            f"{s.feature_count} features shipped, {s.bug_count} open bugs. "
            f"Engineering team size: {s.engineering_team}."
        )

        sales_report = (
            f"Q{s.quarter} Sales Report: {s.customer_count} customers, "
            f"satisfaction {s.customer_satisfaction:.2f}, churn {s.churn_rate*100:.1f}%. "
            f"NPS score {s.nps_score:.0f}. Current price ${s.price:.2f}. "
            f"Market share {s.market_share*100:.1f}%."
        )

        return {
            "ceo": ceo_report,
            "cto": cto_report,
            "sales": sales_report,
        }

    @property
    def ceo_obs_dim(self) -> int:
        return 9

    @property
    def cto_obs_dim(self) -> int:
        return 7

    @property
    def sales_obs_dim(self) -> int:
        return 7

    @property
    def ceo_action_dim(self) -> int:
        return 3  # [rd_pct, marketing_pct, ops_pct]

    @property
    def cto_action_dim(self) -> int:
        return 3  # [new_features, bug_fixes, infra]

    @property
    def sales_action_dim(self) -> int:
        return 3  # [price_delta, discount_rate, outreach_effort]
