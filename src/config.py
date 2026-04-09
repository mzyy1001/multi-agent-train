from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class EnvConfig:
    name: str = "simple_speaker_listener_v4"
    max_cycles: int = 25
    continuous_actions: bool = True


@dataclass
class ModelConfig:
    model_id: str = "HuggingFaceTB/SmolLM2-135M-Instruct"
    hidden_size: int = 576
    dtype: str = "float16"


@dataclass
class CommConfig:
    type: str = "ssr"
    dim: int = 8
    num_symbols: int = 8
    gumbel_tau: float = 1.0
    gumbel_tau_min: float = 0.1


@dataclass
class ModuleConfig:
    projector_hidden: int = 128
    adapter_hidden: int = 128
    action_hidden: int = 64


@dataclass
class TrainingConfig:
    algorithm: str = "ppo"
    total_episodes: int = 50000
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    ppo_epochs: int = 4
    minibatch_size: int = 256
    rollout_episodes: int = 32
    centralized: bool = False


@dataclass
class LogConfig:
    backend: str = "tensorboard"
    log_dir: str = "runs/"
    log_interval: int = 100
    eval_interval: int = 1000
    eval_episodes: int = 50
    save_interval: int = 5000
    wandb_project: str = "multi-agent-ssr"


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints/"
    resume: str | None = None


@dataclass
class Config:
    seed: int = 42
    device: str = "cuda"
    env: EnvConfig = field(default_factory=EnvConfig)
    speaker: ModelConfig = field(default_factory=ModelConfig)
    listener: ModelConfig = field(default_factory=ModelConfig)
    comm: CommConfig = field(default_factory=CommConfig)
    modules: ModuleConfig = field(default_factory=ModuleConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LogConfig = field(default_factory=LogConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)


def _merge_dicts(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged


def _dict_to_config(d: dict[str, Any]) -> Config:
    """Convert a nested dict to a Config dataclass."""
    return Config(
        seed=d.get("seed", 42),
        device=d.get("device", "cuda"),
        env=EnvConfig(**d.get("env", {})),
        speaker=ModelConfig(**d.get("speaker", {})),
        listener=ModelConfig(**d.get("listener", {})),
        comm=CommConfig(**d.get("comm", {})),
        modules=ModuleConfig(**d.get("modules", {})),
        training=TrainingConfig(**d.get("training", {})),
        logging=LogConfig(**d.get("logging", {})),
        checkpoint=CheckpointConfig(**d.get("checkpoint", {})),
    )


def _apply_cli_overrides(cfg_dict: dict, cli_overrides: list[str]) -> dict:
    """Apply dot-notation key=value overrides, e.g. 'comm.dim=16'."""
    for item in cli_overrides:
        key, _, value = item.partition("=")
        if not value:
            continue
        keys = key.split(".")
        d = cfg_dict
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        # Auto-cast value types
        if value.lower() in ("true", "false"):
            d[keys[-1]] = value.lower() == "true"
        elif value.replace(".", "", 1).replace("-", "", 1).replace("e", "", 1).isdigit():
            d[keys[-1]] = float(value) if "." in value or "e" in value else int(value)
        else:
            d[keys[-1]] = value
    return cfg_dict


def load_config(
    base_path: str = "configs/default.yaml",
    overrides: list[str] | None = None,
    cli_overrides: list[str] | None = None,
) -> Config:
    """Load base config YAML, merge overlay YAMLs and CLI overrides, return Config."""
    base_dir = Path(base_path).parent

    with open(base_path) as f:
        cfg_dict = yaml.safe_load(f) or {}

    for override_path in overrides or []:
        full_path = base_dir / override_path
        with open(full_path) as f:
            override_dict = yaml.safe_load(f) or {}
        cfg_dict = _merge_dicts(cfg_dict, override_dict)

    if cli_overrides:
        cfg_dict = _apply_cli_overrides(cfg_dict, cli_overrides)

    return _dict_to_config(cfg_dict)
