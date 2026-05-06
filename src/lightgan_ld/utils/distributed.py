from __future__ import annotations

import os
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.distributed as dist


def distributed_available() -> bool:
    return dist.is_available() and int(os.environ.get("WORLD_SIZE", "1")) > 1


def setup_distributed() -> tuple[bool, int, int, int]:
    if not distributed_available():
        return False, 0, 1, 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        backend = "nccl"
    else:
        backend = "gloo"
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    return True, rank, world, local_rank


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not (dist.is_available() and dist.is_initialized()) or dist.get_rank() == 0


def all_reduce_mean(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        y /= dist.get_world_size()
        return y
    return x

@contextmanager
def main_process_first() -> Iterator[None]:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        dist.barrier()
    yield
    if dist.is_available() and dist.is_initialized() and dist.get_rank() == 0:
        dist.barrier()
