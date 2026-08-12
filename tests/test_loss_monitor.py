from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from loss_monitor import (
    LossWindow,
    TrainStep,
    completed_windows,
    evaluate_alerts,
    read_train_steps,
    write_history_csv,
)


def _train_step(step: int, loss: float, *, epoch: int = 1) -> str:
    return json.dumps(
        {
            "event": "train_step",
            "epoch": epoch,
            "global_step": step,
            "loss": loss,
            "learning_rate": 2e-7,
        }
    )


def test_parser_ignores_noise_and_deduplicates_steps(tmp_path: Path) -> None:
    source = tmp_path / "train.log"
    source.write_text(
        "launcher warning\n"
        + json.dumps({"event": "run_start"})
        + "\n"
        + _train_step(2, 7.2)
        + "\n"
        + _train_step(1, 7.5)
        + "\n"
        + _train_step(2, 6.9)
        + "\n"
    )

    steps = read_train_steps(source)

    assert [(item.global_step, item.loss) for item in steps] == [
        (1, 7.5),
        (2, 6.9),
    ]


def test_completed_windows_use_exact_non_overlapping_ranges(tmp_path: Path) -> None:
    source = tmp_path / "train.log"
    source.write_text(
        "\n".join(_train_step(step, 8.0 - step / 1000) for step in range(1, 251))
        + "\n"
    )

    windows = completed_windows(read_train_steps(source), window_size=100)

    assert [(item.start_step, item.end_step) for item in windows] == [
        (1, 100),
        (101, 200),
    ]
    first = windows[0]
    assert first.epoch == 1
    assert first.last_loss == 7.9
    assert math.isclose(first.mean_loss, 7.9495)
    assert first.min_loss == 7.9
    assert first.max_loss == 7.999
    assert first.learning_rate == 2e-7
    assert first.delta_from_previous is None
    assert first.percent_change is None
    second = windows[1]
    assert math.isclose(second.delta_from_previous, -0.1)
    assert math.isclose(second.percent_change, -0.1 / 7.9495 * 100)


def test_history_csv_is_an_idempotent_snapshot(tmp_path: Path) -> None:
    source = tmp_path / "train.log"
    source.write_text(
        "\n".join(_train_step(step, 7.0) for step in range(1, 201)) + "\n"
    )
    windows = completed_windows(read_train_steps(source), window_size=100)
    history = tmp_path / "history.csv"

    write_history_csv(history, windows)
    write_history_csv(history, windows)

    with history.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["end_step"]) for row in rows] == [100, 200]
    assert not list(tmp_path.glob(".history.csv.*.tmp"))


def test_nonfinite_metrics_remain_visible_for_alerting(tmp_path: Path) -> None:
    source = tmp_path / "train.log"
    source.write_text(_train_step(1, float("nan")) + "\n")

    steps = read_train_steps(source)

    assert len(steps) == 1
    assert math.isnan(steps[0].loss)


def _window(end_step: int, mean_loss: float) -> LossWindow:
    return LossWindow(
        epoch=1,
        start_step=end_step - 99,
        end_step=end_step,
        last_loss=mean_loss,
        mean_loss=mean_loss,
        min_loss=mean_loss,
        max_loss=mean_loss,
        delta_from_previous=None,
        percent_change=None,
        learning_rate=2e-7,
    )


def test_alerts_cover_nonfinite_rise_stall_and_dead_session() -> None:
    alerts = evaluate_alerts(
        windows=[_window(100, 5.0), _window(200, 5.6)],
        steps=[
            TrainStep(
                epoch=1,
                global_step=201,
                loss=float("nan"),
                learning_rate=2e-7,
            )
        ],
        session_alive=False,
        now=400.0,
        last_new_step_at=0.0,
        rise_threshold=0.10,
        stale_seconds=300.0,
    )

    assert {item.split(" ", 1)[0] for item in alerts} == {
        "nonfinite",
        "rise",
        "stalled",
        "session_dead",
    }


def test_once_cli_backfills_without_duplicate_window_messages(tmp_path: Path) -> None:
    source = tmp_path / "train.log"
    source.write_text(
        "\n".join(_train_step(step, 8.0 - step / 1000) for step in range(1, 201))
        + "\n"
    )
    history = tmp_path / "history.csv"
    monitor_log = tmp_path / "monitor.log"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_tmux = fake_bin / "tmux"
    fake_tmux.write_text("#!/bin/sh\nexit 0\n")
    fake_tmux.chmod(0o755)
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "loss_monitor.py"),
        "--once",
        "--source-log",
        str(source),
        "--history-csv",
        str(history),
        "--monitor-log",
        str(monitor_log),
        "--session",
        "test-training",
    ]
    environment = {**os.environ, "PATH": f"{fake_bin}:{os.environ['PATH']}"}

    first = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)
    second = subprocess.run(command, check=False, capture_output=True, text=True, env=environment)

    assert first.returncode == 0, first.stdout + first.stderr
    assert second.returncode == 0, second.stdout + second.stderr
    with history.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["end_step"]) for row in rows] == [100, 200]
    rendered = monitor_log.read_text()
    assert rendered.count("WINDOW end_step=100") == 1
    assert rendered.count("WINDOW end_step=200") == 1
