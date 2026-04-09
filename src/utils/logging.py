from __future__ import annotations

from pathlib import Path

from src.config import LogConfig


class Logger:
    """Unified logging to tensorboard and/or wandb."""

    def __init__(self, config: LogConfig, run_name: str = "default"):
        self.config = config
        self.backend = config.backend
        self._tb_writer = None
        self._wandb_run = None

        if self.backend in ("tensorboard", "both"):
            from torch.utils.tensorboard import SummaryWriter
            log_path = Path(config.log_dir) / run_name
            self._tb_writer = SummaryWriter(str(log_path))

        if self.backend in ("wandb", "both"):
            import wandb
            self._wandb_run = wandb.init(
                project=config.wandb_project,
                name=run_name,
            )

    def log(
        self,
        metrics: dict[str, float],
        step: int,
        prefix: str = "",
    ):
        for key, value in metrics.items():
            tag = f"{prefix}{key}" if prefix else key
            if self._tb_writer:
                self._tb_writer.add_scalar(tag, value, step)
            if self._wandb_run:
                import wandb
                wandb.log({tag: value}, step=step)

    def close(self):
        if self._tb_writer:
            self._tb_writer.close()
        if self._wandb_run:
            import wandb
            wandb.finish()
