from __future__ import annotations

import copy
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch


FORMAT_VERSION = 1
CHECKPOINT_KIND = "focat_training"


@dataclass(frozen=True)
class ResumeState:
    epoch: int
    step_in_epoch: int
    global_step: int
    aggregate_k_gradients: int
    config: dict[str, Any]
    seed: int
    loss: float
    learning_rate: float
    source: dict[str, Any]
    started_at: str


def capture_rng_state() -> dict[str, Any]:
    cuda_state = None
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        cuda_state = torch.cuda.get_rng_state()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": cuda_state,
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state(state["cuda"])


def _unwrapped_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if hasattr(model, "module") else model


def checkpoint_payload(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    epoch: int,
    step_in_epoch: int,
    global_step: int,
    config: dict[str, Any],
    seed: int,
    world_size: int,
    rng_states: list[dict[str, Any]],
    loss: float,
    learning_rate: float,
    aggregate_k_gradients: int,
    source: dict[str, Any],
    started_at: str,
) -> dict[str, Any]:
    if global_step < 1:
        raise ValueError("global_step must be positive for a training checkpoint")
    if len(rng_states) != world_size:
        raise ValueError("one RNG state is required for every distributed rank")
    return {
        "format_version": FORMAT_VERSION,
        "kind": CHECKPOINT_KIND,
        "model_state": _unwrapped_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "epoch": int(epoch),
        "step_in_epoch": int(step_in_epoch),
        "global_step": int(global_step),
        "config": copy.deepcopy(config),
        "seed": int(seed),
        "world_size": int(world_size),
        "rng_states": rng_states,
        "loss": float(loss),
        "learning_rate": float(learning_rate),
        "aggregate_k_gradients": int(aggregate_k_gradients),
        "source": copy.deepcopy(source),
        "started_at": started_at,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }


def _step_checkpoints(checkpoint_dir: Path) -> list[Path]:
    paths = []
    for path in checkpoint_dir.glob("step_????????.pt"):
        digits = path.stem.removeprefix("step_")
        if len(digits) == 8 and digits.isdigit():
            paths.append(path)
    return sorted(paths)


def save_training_checkpoint(
    payload: dict[str, Any],
    checkpoint_dir: str | Path,
    *,
    keep_last: int,
) -> Path:
    if keep_last < 1:
        raise ValueError("keep_last must be at least 1")
    global_step = int(payload["global_step"])
    if global_step < 1:
        raise ValueError("checkpoint global_step must be positive")
    directory = Path(checkpoint_dir)
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = directory / f"step_{global_step:08d}.pt"
    temporary = directory / f".{checkpoint.name}.{os.getpid()}.tmp"
    latest = directory / "latest.pt"
    latest_temporary = directory / f".latest.pt.{os.getpid()}.tmp"
    try:
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint)
        if latest_temporary.exists():
            latest_temporary.unlink()
        os.link(checkpoint, latest_temporary)
        os.replace(latest_temporary, latest)
    finally:
        if temporary.exists():
            temporary.unlink()
        if latest_temporary.exists():
            latest_temporary.unlink()
    old_checkpoints = _step_checkpoints(directory)[:-keep_last]
    for old_checkpoint in old_checkpoints:
        old_checkpoint.unlink()
    return checkpoint


def _load_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("training resume requires a dictionary checkpoint")
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format version: {payload.get('format_version')!r}"
        )
    if payload.get("kind") != CHECKPOINT_KIND:
        raise ValueError(f"unsupported checkpoint kind: {payload.get('kind')!r}")
    return payload


def load_training_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    expected_world_size: int,
    rank: int,
    restore_rng: bool = True,
) -> ResumeState:
    payload = _load_payload(path)
    saved_world_size = int(payload["world_size"])
    if saved_world_size != expected_world_size:
        raise ValueError(
            f"checkpoint world size {saved_world_size} does not match current "
            f"world size {expected_world_size}"
        )
    rng_states = payload["rng_states"]
    if len(rng_states) != saved_world_size or not 0 <= rank < saved_world_size:
        raise ValueError("checkpoint RNG state does not match distributed ranks")
    _unwrapped_model(model).load_state_dict(payload["model_state"], strict=True)
    optimizer.load_state_dict(payload["optimizer_state"])
    scheduler.load_state_dict(payload["scheduler_state"])
    if restore_rng:
        restore_rng_state(rng_states[rank])
    return ResumeState(
        epoch=int(payload["epoch"]),
        step_in_epoch=int(payload["step_in_epoch"]),
        global_step=int(payload["global_step"]),
        aggregate_k_gradients=int(payload["aggregate_k_gradients"]),
        config=copy.deepcopy(payload["config"]),
        seed=int(payload["seed"]),
        loss=float(payload["loss"]),
        learning_rate=float(payload["learning_rate"]),
        source=copy.deepcopy(payload.get("source", {})),
        started_at=str(payload.get("started_at", "")),
    )
