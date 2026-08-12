from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class TrainStep:
    epoch: int
    global_step: int
    loss: float
    learning_rate: float


@dataclass(frozen=True)
class LossWindow:
    epoch: int
    start_step: int
    end_step: int
    last_loss: float
    mean_loss: float
    min_loss: float
    max_loss: float
    delta_from_previous: float | None
    percent_change: float | None
    learning_rate: float


CSV_FIELDS = tuple(LossWindow.__dataclass_fields__)


def read_train_steps(path: Path) -> list[TrainStep]:
    by_step: dict[int, TrainStep] = {}
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            try:
                record = json.loads(line)
                if record.get("event") != "train_step":
                    continue
                step = TrainStep(
                    epoch=int(record["epoch"]),
                    global_step=int(record["global_step"]),
                    loss=float(record["loss"]),
                    learning_rate=float(record["learning_rate"]),
                )
            except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError):
                continue
            by_step[step.global_step] = step
    return [by_step[key] for key in sorted(by_step)]


def completed_windows(
    steps: list[TrainStep],
    window_size: int,
) -> list[LossWindow]:
    if window_size < 1:
        raise ValueError("window_size must be positive")
    by_step = {item.global_step: item for item in steps}
    if not by_step:
        return []
    windows: list[LossWindow] = []
    last_complete_end = max(by_step) // window_size * window_size
    for end_step in range(window_size, last_complete_end + 1, window_size):
        start_step = end_step - window_size + 1
        try:
            items = [by_step[value] for value in range(start_step, end_step + 1)]
        except KeyError:
            continue
        losses = [item.loss for item in items]
        mean_loss = math.fsum(losses) / window_size
        previous_mean = windows[-1].mean_loss if windows else None
        delta = None if previous_mean is None else mean_loss - previous_mean
        percent = (
            None
            if previous_mean is None or previous_mean == 0
            else delta / previous_mean * 100
        )
        windows.append(
            LossWindow(
                epoch=items[-1].epoch,
                start_step=start_step,
                end_step=end_step,
                last_loss=items[-1].loss,
                mean_loss=mean_loss,
                min_loss=min(losses),
                max_loss=max(losses),
                delta_from_previous=delta,
                percent_change=percent,
                learning_rate=items[-1].learning_rate,
            )
        )
    return windows


def write_history_csv(path: Path, windows: list[LossWindow]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            for window in windows:
                writer.writerow(asdict(window))
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
