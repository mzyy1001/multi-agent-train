from __future__ import annotations

import torch
import torch.nn as nn
from torch.distributions import Normal


class ContinuousActionHead(nn.Module):
    """Policy + value heads for continuous actions.

    Uses tanh-squashed Gaussian for reparameterizable sampling,
    keeping gradients flowing through the message.
    """

    LOG_STD_MIN = -5.0
    LOG_STD_MAX = 2.0

    def __init__(self, input_dim: int, action_dim: int, hidden_dim: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (mean, log_std, value)."""
        features = self.shared(h)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        value = self.value_head(h).squeeze(-1)
        return mean, log_std, value

    def get_action(
        self,
        h: torch.Tensor,
        deterministic: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Sample action, return action, log_prob, value, entropy."""
        mean, log_std, value = self.forward(h)
        std = log_std.exp()
        dist = Normal(mean, std)

        if deterministic:
            raw_action = mean
        else:
            raw_action = dist.rsample()  # reparameterization trick

        # Tanh squash to [0, 1] for PettingZoo continuous actions
        action = torch.sigmoid(raw_action)

        # Log prob with tanh correction
        log_prob = dist.log_prob(raw_action).sum(-1)
        # Correction for sigmoid transform: log |d(sigmoid)/d(raw)| = log(sigmoid * (1 - sigmoid))
        log_prob -= torch.log(action * (1 - action) + 1e-6).sum(-1)

        entropy = dist.entropy().sum(-1)

        return {
            "action": action,
            "log_prob": log_prob,
            "value": value,
            "entropy": entropy,
            "mean": mean,
        }

    def evaluate_action(
        self,
        h: torch.Tensor,
        raw_action: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Evaluate log_prob and value for a given raw (pre-sigmoid) action."""
        mean, log_std, value = self.forward(h)
        std = log_std.exp()
        dist = Normal(mean, std)

        action = torch.sigmoid(raw_action)
        log_prob = dist.log_prob(raw_action).sum(-1)
        log_prob -= torch.log(action * (1 - action) + 1e-6).sum(-1)
        entropy = dist.entropy().sum(-1)

        return {
            "log_prob": log_prob,
            "value": value,
            "entropy": entropy,
        }


class ValueOnlyHead(nn.Module):
    """Value head only, for the speaker (which doesn't produce movement actions)."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """Returns value estimate, shape (batch,)."""
        return self.net(h).squeeze(-1)
