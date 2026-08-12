from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import time


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
ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE_LOG = ROOT / "logs" / "focat_pretrain.log"
DEFAULT_HISTORY_CSV = ROOT / "logs" / "focat_loss_history.csv"
DEFAULT_MONITOR_LOG = ROOT / "logs" / "focat_loss_monitor.log"


def _parse_train_step(line: str) -> TrainStep | None:
    try:
        record = json.loads(line)
        if record.get("event") != "train_step":
            return None
        return TrainStep(
            epoch=int(record["epoch"]),
            global_step=int(record["global_step"]),
            loss=float(record["loss"]),
            learning_rate=float(record["learning_rate"]),
        )
    except (json.JSONDecodeError, AttributeError, KeyError, TypeError, ValueError):
        return None


def read_train_steps(path: Path) -> list[TrainStep]:
    by_step: dict[int, TrainStep] = {}
    with Path(path).open(errors="replace") as handle:
        for line in handle:
            step = _parse_train_step(line)
            if step is None:
                continue
            by_step[step.global_step] = step
    return [by_step[key] for key in sorted(by_step)]


def _make_window(
    items: list[TrainStep],
    previous_mean: float | None,
) -> LossWindow:
    losses = [item.loss for item in items]
    mean_loss = math.fsum(losses) / len(losses)
    delta = None if previous_mean is None else mean_loss - previous_mean
    percent = (
        None
        if previous_mean is None or previous_mean == 0
        else delta / previous_mean * 100
    )
    return LossWindow(
        epoch=items[-1].epoch,
        start_step=items[0].global_step,
        end_step=items[-1].global_step,
        last_loss=items[-1].loss,
        mean_loss=mean_loss,
        min_loss=min(losses),
        max_loss=max(losses),
        delta_from_previous=delta,
        percent_change=percent,
        learning_rate=items[-1].learning_rate,
    )


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
        previous_mean = windows[-1].mean_loss if windows else None
        windows.append(_make_window(items, previous_mean))
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


def _optional_float(value: str | None) -> float | None:
    return None if value in (None, "", "None") else float(value)


def read_history_csv(path: Path) -> list[LossWindow]:
    source = Path(path)
    if not source.is_file():
        return []
    with source.open(newline="") as handle:
        return [
            LossWindow(
                epoch=int(row["epoch"]),
                start_step=int(row["start_step"]),
                end_step=int(row["end_step"]),
                last_loss=float(row["last_loss"]),
                mean_loss=float(row["mean_loss"]),
                min_loss=float(row["min_loss"]),
                max_loss=float(row["max_loss"]),
                delta_from_previous=_optional_float(row["delta_from_previous"]),
                percent_change=_optional_float(row["percent_change"]),
                learning_rate=float(row["learning_rate"]),
            )
            for row in csv.DictReader(handle)
        ]


class _TailReader:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.offset = 0
        self.remainder = b""

    def read(self) -> list[TrainStep]:
        size = self.path.stat().st_size
        if size < self.offset:
            self.offset = 0
            self.remainder = b""
        with self.path.open("rb") as handle:
            handle.seek(self.offset)
            data = handle.read()
            self.offset = handle.tell()
        combined = self.remainder + data
        lines = combined.split(b"\n")
        self.remainder = lines.pop()
        by_step: dict[int, TrainStep] = {}
        for raw_line in lines:
            step = _parse_train_step(raw_line.decode(errors="replace"))
            if step is not None:
                by_step[step.global_step] = step
        return [by_step[key] for key in sorted(by_step)]


class _Accumulator:
    def __init__(self, window_size: int, windows: list[LossWindow]):
        self.window_size = window_size
        self.windows = list(windows)
        self.pending: dict[int, TrainStep] = {}

    @property
    def latest_step(self) -> int:
        values = [0]
        if self.windows:
            values.append(self.windows[-1].end_step)
        if self.pending:
            values.append(max(self.pending))
        return max(values)

    def add(self, steps: list[TrainStep]) -> list[LossWindow]:
        completed_end = self.windows[-1].end_step if self.windows else 0
        for step in steps:
            if step.global_step > completed_end:
                self.pending[step.global_step] = step
        created = []
        next_end = completed_end + self.window_size
        while all(
            value in self.pending
            for value in range(next_end - self.window_size + 1, next_end + 1)
        ):
            start = next_end - self.window_size + 1
            items = [self.pending.pop(value) for value in range(start, next_end + 1)]
            previous_mean = self.windows[-1].mean_loss if self.windows else None
            window = _make_window(items, previous_mean)
            self.windows.append(window)
            created.append(window)
            next_end += self.window_size
        return created


def evaluate_alerts(
    *,
    windows: list[LossWindow],
    steps: list[TrainStep],
    session_alive: bool,
    now: float,
    last_new_step_at: float,
    rise_threshold: float,
    stale_seconds: float,
) -> list[str]:
    alerts = []
    nonfinite = next(
        (
            item
            for item in steps
            if not math.isfinite(item.loss) or not math.isfinite(item.learning_rate)
        ),
        None,
    )
    if nonfinite is not None:
        alerts.append(f"nonfinite global_step={nonfinite.global_step}")
    if len(windows) >= 2:
        previous = windows[-2].mean_loss
        current = windows[-1].mean_loss
        if (
            math.isfinite(previous)
            and math.isfinite(current)
            and previous != 0
            and (current - previous) / abs(previous) >= rise_threshold
        ):
            alerts.append(
                f"rise end_step={windows[-1].end_step} previous={previous:.6f} "
                f"current={current:.6f}"
            )
    if now - last_new_step_at >= stale_seconds:
        alerts.append(f"stalled seconds={now - last_new_step_at:.0f}")
    if not session_alive:
        alerts.append("session_dead")
    return alerts


def _session_alive(session: str) -> bool:
    try:
        result = subprocess.run(
            ["tmux", "has-session", "-t", session],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _append_monitor_log(path: Path, message: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat()
    with destination.open("a") as handle:
        handle.write(f"{timestamp} {message}\n")


def _format_window(window: LossWindow) -> str:
    percent = (
        "n/a" if window.percent_change is None else f"{window.percent_change:+.3f}%"
    )
    return (
        f"WINDOW end_step={window.end_step} epoch={window.epoch} "
        f"mean={window.mean_loss:.6f} delta_pct={percent} "
        f"last={window.last_loss:.6f} min={window.min_loss:.6f} "
        f"max={window.max_loss:.6f} lr={window.learning_rate:.12g}"
    )


def run_monitor(
    *,
    source_log: Path,
    history_csv: Path,
    monitor_log: Path,
    session: str,
    window_size: int,
    poll_seconds: float,
    stale_seconds: float,
    rise_threshold: float,
    once: bool,
) -> None:
    if window_size < 1 or poll_seconds <= 0 or stale_seconds <= 0:
        raise ValueError("window and timing values must be positive")
    if rise_threshold <= 0:
        raise ValueError("rise_threshold must be positive")
    if not Path(source_log).is_file():
        raise FileNotFoundError(f"training log is missing: {source_log}")

    existing_windows = read_history_csv(history_csv)
    accumulator = _Accumulator(window_size, existing_windows)
    reader = _TailReader(source_log)
    last_new_step_at = time.monotonic()
    last_heartbeat_at = 0.0
    active_alerts: set[str] = set()
    _append_monitor_log(
        monitor_log,
        f"START session={session} window_size={window_size} source={Path(source_log).resolve()}",
    )

    while True:
        now = time.monotonic()
        previous_latest = accumulator.latest_step
        new_steps = reader.read()
        created = accumulator.add(new_steps)
        if accumulator.latest_step > previous_latest:
            last_new_step_at = now
        if created or not Path(history_csv).is_file():
            write_history_csv(history_csv, accumulator.windows)
        for window in created:
            _append_monitor_log(monitor_log, _format_window(window))

        alerts = set(
            evaluate_alerts(
                windows=accumulator.windows,
                steps=new_steps,
                session_alive=_session_alive(session),
                now=now,
                last_new_step_at=last_new_step_at,
                rise_threshold=rise_threshold,
                stale_seconds=stale_seconds,
            )
        )
        for alert in sorted(alerts - active_alerts):
            _append_monitor_log(monitor_log, f"ALERT {alert}")
        active_alerts = alerts

        if last_heartbeat_at == 0.0 or now - last_heartbeat_at >= stale_seconds:
            recent_mean = (
                "n/a"
                if not accumulator.windows
                else f"{accumulator.windows[-1].mean_loss:.6f}"
            )
            _append_monitor_log(
                monitor_log,
                f"HEARTBEAT global_step={accumulator.latest_step} recent_mean={recent_mean}",
            )
            last_heartbeat_at = now
        if once:
            return
        time.sleep(poll_seconds)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor FoCAT pretraining loss")
    parser.add_argument("--source-log", type=Path, default=DEFAULT_SOURCE_LOG)
    parser.add_argument("--history-csv", type=Path, default=DEFAULT_HISTORY_CSV)
    parser.add_argument("--monitor-log", type=Path, default=DEFAULT_MONITOR_LOG)
    parser.add_argument("--session", default="focat_pretrain")
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--stale-seconds", type=float, default=300.0)
    parser.add_argument("--rise-threshold", type=float, default=0.10)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_monitor(**vars(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
