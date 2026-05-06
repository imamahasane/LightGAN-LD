from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


def unwrap(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: str | Path, *, generator: nn.Module, discriminator: nn.Module | None = None, encoder: nn.Module | None = None, opt_g: Any = None, opt_d: Any = None, scaler: Any = None, epoch: int = 0, step: int = 0, best_metric: float | None = None, cfg: dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"generator": unwrap(generator).state_dict(), "epoch": epoch, "step": step, "best_metric": best_metric, "cfg": cfg or {}}
    if discriminator is not None:
        state["discriminator"] = unwrap(discriminator).state_dict()
    if encoder is not None:
        state["encoder"] = unwrap(encoder).state_dict()
    if opt_g is not None:
        state["opt_g"] = opt_g.state_dict()
    if opt_d is not None:
        state["opt_d"] = opt_d.state_dict()
    if scaler is not None:
        state["scaler"] = scaler.state_dict()
    torch.save(state, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    return torch.load(path, map_location=map_location)


def load_model_weights(model: nn.Module, state: dict[str, Any], key: str = "generator", strict: bool = True) -> None:
    if key in state:
        weights = state[key]
    elif "G" in state and key == "generator":
        weights = state["G"]
    else:
        weights = state
    unwrap(model).load_state_dict(weights, strict=strict)
