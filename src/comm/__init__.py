from __future__ import annotations

from src.config import CommConfig

from .base import CommChannel
from .continuous import ContinuousChannel
from .discrete import DiscreteChannel
from .none import NoChannel
from .ssr import SSRChannel


def build_comm_channel(cfg: CommConfig, input_dim: int) -> CommChannel:
    """Factory: create a communication channel from config."""
    if cfg.type == "ssr":
        return SSRChannel(input_dim, cfg.dim)
    elif cfg.type == "discrete":
        return DiscreteChannel(input_dim, cfg.num_symbols, cfg.gumbel_tau)
    elif cfg.type == "continuous":
        return ContinuousChannel(input_dim, cfg.dim)
    elif cfg.type == "none":
        return NoChannel(cfg.dim or 1)
    else:
        raise ValueError(f"Unknown comm type: {cfg.type}")
