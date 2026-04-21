"""Configurable Market Simulation Environment for Multi-Agent Company RL.

Parameterizable scenarios for training and testing generalization:
- Market conditions (growth, volatility, competition)
- Company starting state (startup vs established)
- Crisis events (recession, disruption, talent war)
- Difficulty presets (easy/medium/hard/nightmare)

Each scenario tests different aspects of multi-agent coordination.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ScenarioConfig:
    """Configurable scenario parameters for testing generalization."""
    name: str = "default"

    # Game length
    n_quarters: int = 8

    # Market
    initial_market_size: float = 1000.0
    market_growth_mean: float = 0.05
    market_volatility: float = 0.05
    demand_elasticity: float = 1.5

    # Competition
    n_competitors: int = 1
    competitor_quality_init: float = 0.5
    competitor_aggressiveness: float = 0.05
    competitor_price: float = 10.0

    # Company starting state
    initial_quality: float = 0.4
    initial_tech_debt: float = 0.15
    initial_cash: float = 200.0
    initial_price: float = 10.0
    initial_customers: int = 100
    initial_eng_team: int = 5
    initial_sales_team: int = 3

    # Event probabilities
    crash_prob: float = 0.05
    new_competitor_prob: float = 0.05
    viral_prob: float = 0.05
    talent_loss_prob: float = 0.03
    regulation_prob: float = 0.02

    # Costs
    salary_per_person: float = 15.0
    hiring_cost: float = 30.0

    # Reward shaping
    bankruptcy_penalty: float = -10.0
    survival_bonus: float = 2.0

    @staticmethod
    def easy():
        return ScenarioConfig(
            name="easy", n_quarters=6, market_growth_mean=0.08,
            market_volatility=0.02, competitor_aggressiveness=0.02,
            initial_quality=0.6, initial_cash=400, initial_customers=200,
            crash_prob=0.01, new_competitor_prob=0.01,
        )

    @staticmethod
    def medium():
        return ScenarioConfig(name="medium")

    @staticmethod
    def hard():
        return ScenarioConfig(
            name="hard", n_quarters=10, market_growth_mean=0.02,
            market_volatility=0.08, competitor_aggressiveness=0.08,
            competitor_quality_init=0.6, initial_quality=0.3,
            initial_cash=150, crash_prob=0.08, new_competitor_prob=0.08,
        )

    @staticmethod
    def startup():
        """Early-stage startup: low resources, high growth potential."""
        return ScenarioConfig(
            name="startup", n_quarters=12, initial_market_size=500,
            market_growth_mean=0.10, initial_quality=0.2, initial_tech_debt=0.05,
            initial_cash=100, initial_customers=20, initial_eng_team=3,
            initial_sales_team=1, initial_price=15.0,
        )

    @staticmethod
    def recession():
        """Economic downturn: shrinking market, cost pressure."""
        return ScenarioConfig(
            name="recession", n_quarters=8, market_growth_mean=-0.03,
            market_volatility=0.10, competitor_aggressiveness=0.10,
            initial_cash=150, crash_prob=0.15, talent_loss_prob=0.08,
        )

    @staticmethod
    def disruption():
        """Competitor disruption: strong new entrant."""
        return ScenarioConfig(
            name="disruption", n_quarters=8, competitor_quality_init=0.7,
            competitor_aggressiveness=0.12, n_competitors=2,
            new_competitor_prob=0.10,
        )


@dataclass
class MarketState:
    """Full market state."""
    quarter: int = 0
    product_quality: float = 0.5
    tech_debt: float = 0.1
    feature_count: int = 3
    bug_count: int = 2
    market_size: float = 1000.0
    market_share: float = 0.1
    competitor_quality: float = 0.5
    market_growth: float = 0.05
    customer_count: int = 100
    customer_satisfaction: float = 0.6
    churn_rate: float = 0.05
    nps_score: float = 30.0
    revenue: float = 100.0
    costs: float = 80.0
    profit: float = 20.0
    cash: float = 200.0
    price: float = 10.0
    engineering_team: int = 5
    sales_team: int = 3
    quarterly_profits: list = field(default_factory=list)
    events_log: list = field(default_factory=list)


class MarketSimEnv:
    """Multi-quarter business simulation with configurable scenarios."""

    # Observation dimensions
    CEO_OBS_DIM = 9
    CTO_OBS_DIM = 7
    SALES_OBS_DIM = 7
    CEO_ACT_DIM = 3
    CTO_ACT_DIM = 3
    SALES_ACT_DIM = 3

    def __init__(self, config: ScenarioConfig = None, seed: int = 42):
        self.config = config or ScenarioConfig.medium()
        self.rng = np.random.RandomState(seed)
        self.state: Optional[MarketState] = None

    def reset(self, seed: int = None, config: ScenarioConfig = None) -> dict[str, np.ndarray]:
        """Reset environment. Optionally change scenario config."""
        if config is not None:
            self.config = config
        if seed is not None:
            self.rng = np.random.RandomState(seed)

        c = self.config
        noise = lambda s: self.rng.uniform(-s, s)

        self.state = MarketState(
            quarter=0,
            product_quality=np.clip(c.initial_quality + noise(0.05), 0.1, 0.9),
            tech_debt=np.clip(c.initial_tech_debt + noise(0.05), 0.0, 0.5),
            market_size=c.initial_market_size + noise(50),
            market_growth=c.market_growth_mean,
            competitor_quality=c.competitor_quality_init + noise(0.05),
            cash=c.initial_cash + noise(20),
            price=c.initial_price + noise(1),
            engineering_team=c.initial_eng_team,
            sales_team=c.initial_sales_team,
            customer_count=c.initial_customers + int(noise(10)),
            customer_satisfaction=0.5 + noise(0.1),
        )
        s = self.state
        s.market_share = s.customer_count * s.price / max(1, s.market_size)
        s.revenue = s.customer_count * s.price
        s.costs = (s.engineering_team + s.sales_team) * c.salary_per_person
        s.profit = s.revenue - s.costs

        return self._get_observations()

    def step(self, ceo_action, cto_action, sales_action):
        """Execute one quarter. Returns (obs, reward, done, info)."""
        s = self.state
        c = self.config
        s.quarter += 1

        # Normalize actions
        ceo_a = np.clip(ceo_action, 0.05, 1.0)
        ceo_a = ceo_a / (ceo_a.sum() + 1e-8)
        rd_pct, mkt_pct, ops_pct = ceo_a

        cto_a = np.clip(cto_action, 0.05, 1.0)
        cto_a = cto_a / (cto_a.sum() + 1e-8)
        feat_f, bug_f, infra_f = cto_a

        price_delta = np.clip(sales_action[0], -0.3, 0.3)
        discount = np.clip(sales_action[1], 0.0, 0.5)
        outreach = np.clip(sales_action[2], 0.0, 1.0)

        # Budget
        budget = max(0, s.revenue * 0.7)
        rd_budget = budget * rd_pct
        mkt_budget = budget * mkt_pct

        # === Engineering ===
        eff = 1.0 - s.tech_debt  # effectiveness reduced by tech debt
        quality_gain = feat_f * rd_budget / 150.0 * eff
        s.product_quality = np.clip(s.product_quality + quality_gain * 0.08, 0, 1)
        s.feature_count += max(0, int(quality_gain * 3))

        bugs_fixed = int(bug_f * s.engineering_team * 1.5)
        s.bug_count = max(0, s.bug_count - bugs_fixed)
        s.tech_debt = np.clip(s.tech_debt - bug_f * 0.04 - infra_f * 0.03 + feat_f * 0.025, 0, 1)
        s.bug_count += self.rng.poisson(s.tech_debt * 2.5)

        # === Pricing ===
        s.price = np.clip(s.price * (1 + price_delta), 1.0, 50.0)
        eff_price = s.price * (1 - discount)

        # === Customers ===
        q_factor = s.product_quality * max(0, 1 - s.bug_count / 15.0)
        p_fair = max(0, 1.0 - abs(eff_price - c.competitor_price) / 15.0)
        s.customer_satisfaction = np.clip(
            0.3 * q_factor + 0.2 * p_fair + 0.5 * s.customer_satisfaction + self.rng.normal(0, 0.02),
            0, 1)

        s.churn_rate = np.clip(0.15 - s.customer_satisfaction * 0.12, 0.01, 0.3)
        churned = int(s.customer_count * s.churn_rate)

        mkt_reach = mkt_budget / 60.0 * outreach
        organic = s.customer_satisfaction * max(0, s.nps_score) / 100.0
        new_cust = int(mkt_reach * 4 + organic * s.customer_count * 0.015)
        new_cust = max(0, new_cust + self.rng.randint(-3, 8))
        s.customer_count = max(1, s.customer_count - churned + new_cust)

        s.nps_score = np.clip(
            s.nps_score + (s.customer_satisfaction - 0.5) * 8 + self.rng.normal(0, 2), -100, 100)

        # === Market ===
        s.market_size *= (1 + s.market_growth + self.rng.normal(0, c.market_volatility))
        s.market_size = max(100, s.market_size)
        s.market_share = np.clip(s.customer_count * eff_price / max(1, s.market_size), 0, 0.8)

        s.competitor_quality = np.clip(
            s.competitor_quality + self.rng.normal(c.competitor_aggressiveness * 0.5, 0.02), 0, 1)
        if s.competitor_quality > s.product_quality + 0.1:
            s.market_share *= 0.95

        # === Events ===
        event = None
        r = self.rng.random()
        cum = 0
        for prob, ev_name, ev_fn in [
            (c.crash_prob, "market_crash", lambda: setattr(s, 'market_size', s.market_size * 0.8)),
            (c.new_competitor_prob, "new_competitor", lambda: setattr(s, 'competitor_quality', min(1, s.competitor_quality + 0.12))),
            (c.viral_prob, "viral_growth", lambda: setattr(s, 'customer_count', int(s.customer_count * 1.25))),
            (c.talent_loss_prob, "talent_loss", lambda: setattr(s, 'engineering_team', max(2, s.engineering_team - 1))),
            (c.regulation_prob, "regulation", lambda: setattr(s, 'costs', s.costs + 30)),
        ]:
            cum += prob
            if r < cum:
                ev_fn()
                event = ev_name
                s.events_log.append((s.quarter, event))
                break

        # === Financials ===
        s.revenue = s.customer_count * eff_price
        personnel = (s.engineering_team + s.sales_team) * c.salary_per_person
        s.costs = personnel + budget * ops_pct * 0.3
        s.profit = s.revenue - s.costs
        s.cash += s.profit
        s.quarterly_profits.append(s.profit)

        # Hiring
        if ops_pct > 0.3 and s.cash > c.hiring_cost * 2:
            if self.rng.random() < ops_pct * 0.4:
                s.engineering_team += 1
                s.cash -= c.hiring_cost
            if self.rng.random() < ops_pct * 0.25:
                s.sales_team += 1
                s.cash -= c.hiring_cost

        done = s.quarter >= c.n_quarters or s.cash < -50
        reward = s.profit / 100.0
        if s.cash < -50:
            reward += c.bankruptcy_penalty
        if done and s.cash > 0:
            reward += c.survival_bonus

        info = {
            "quarter": s.quarter, "profit": s.profit, "revenue": s.revenue,
            "costs": s.costs, "cash": s.cash, "market_share": s.market_share,
            "customer_count": s.customer_count, "product_quality": s.product_quality,
            "tech_debt": s.tech_debt, "satisfaction": s.customer_satisfaction,
            "event": event, "cumulative_profit": sum(s.quarterly_profits),
            "scenario": c.name,
        }
        return self._get_observations(), reward, done, info

    def _get_observations(self):
        s = self.state
        ceo = np.array([
            s.market_size / 2000, s.market_share, s.market_growth,
            s.competitor_quality, s.revenue / 500, s.costs / 500,
            s.profit / 200, s.cash / 500, s.quarter / self.config.n_quarters,
        ], dtype=np.float32)
        cto = np.array([
            s.product_quality, s.tech_debt, s.feature_count / 20,
            s.bug_count / 20, s.engineering_team / 10,
            s.quarter / self.config.n_quarters, s.revenue / 500,
        ], dtype=np.float32)
        sales = np.array([
            s.customer_satisfaction, s.churn_rate, s.nps_score / 100,
            s.customer_count / 500, s.price / 20, s.market_share,
            s.quarter / self.config.n_quarters,
        ], dtype=np.float32)
        return {"ceo": ceo, "cto": cto, "sales": sales}

    def get_text_report(self):
        s = self.state
        return {
            "ceo": f"Q{s.quarter} CEO: Mkt ${s.market_size:.0f}, share {s.market_share*100:.1f}%, "
                   f"growth {s.market_growth*100:.1f}%, competitor {s.competitor_quality:.2f}. "
                   f"Rev ${s.revenue:.0f}, cost ${s.costs:.0f}, profit ${s.profit:.0f}, cash ${s.cash:.0f}.",
            "cto": f"Q{s.quarter} CTO: Quality {s.product_quality:.2f}, debt {s.tech_debt:.2f}, "
                   f"{s.feature_count} features, {s.bug_count} bugs, team {s.engineering_team}.",
            "sales": f"Q{s.quarter} Sales: {s.customer_count} customers, sat {s.customer_satisfaction:.2f}, "
                     f"churn {s.churn_rate*100:.1f}%, NPS {s.nps_score:.0f}, price ${s.price:.1f}.",
        }
