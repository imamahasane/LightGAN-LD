from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from torch.utils.tensorboard import SummaryWriter


class ExperimentLogger:
    def __init__(self, log_dir: str, use_wandb: bool = False, wandb_project: str = "lightgan-ld", config: dict[str, Any] | None = None):
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.tb = SummaryWriter(log_dir)
        self.wandb = None
        if use_wandb:
            try:
                import wandb
                self.wandb = wandb
                wandb.init(project=wandb_project, config=config or {}, dir=str(Path(log_dir).parent))
            except Exception as exc:  # pragma: no cover
                logging.warning("WandB disabled: %s", exc)

    def scalar(self, name: str, value: float, step: int) -> None:
        self.tb.add_scalar(name, value, step)
        if self.wandb is not None:
            self.wandb.log({name: value, "step": step})

    def image(self, name: str, tensor, step: int) -> None:
        self.tb.add_image(name, tensor, step)
        if self.wandb is not None:
            self.wandb.log({name: self.wandb.Image(tensor.detach().cpu()), "step": step})

    def flush(self) -> None:
        self.tb.flush()

    def close(self) -> None:
        self.tb.close()
        if self.wandb is not None:
            self.wandb.finish()


def setup_text_logger(log_file: str | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)
