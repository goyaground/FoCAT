# FoCAT Checkpoint Cadence Transition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Safely diagnose the stalled four-GPU FoCAT step and resume full pre-training from the last verified checkpoint with a 2,000-optimizer-step checkpoint interval and three retained checkpoints.

**Architecture:** Preserve the production checkpoint as the source of truth, stop the currently stalled DDP job only after validating it, and use a small watchdog wrapper to replay the exact saved RNG state on GPUs 0-3. The replay is a diagnosis gate: resume production unchanged if the suspect step completes, or stop and design one evidence-backed runtime fix if stack dumps reproduce the stall.

**Tech Stack:** Python 3.10, PyTorch 2.7, `torchrun`/DDP, `faulthandler`, pytest, tmux, NVIDIA tooling.

## Global Constraints

- Do not change FoCAT model architecture, repository synthetic-prior probabilities, sampled prior distributions, optimizer, scheduler, seed 42, global batch size 8, or adaptive gradient accumulation.
- Use physical GPUs 0, 1, 2, and 3 for both deterministic replay and resumed production.
- Treat `checkpoints/focat/latest.pt` at global step 209,000 as immutable until a newer production checkpoint is atomically saved.
- Keep `--keep-last 3`; change only the production checkpoint interval from 1,000 to 2,000 optimizer steps.
- Append to `logs/focat_pretrain.log`; do not truncate historical loss or error evidence.
- Do not restart production repeatedly if the deterministic replay reproduces the stall.

---

### Task 1: Validate the Recovery Point and Stop the Stalled Job

**Files:**
- Read: `checkpoints/focat/latest.pt`
- Record: `logs/focat_stall_snapshot_20260813.log`
- Preserve: `logs/focat_pretrain.log`

**Interfaces:**
- Consumes: `ticl.checkpointing._load_payload(path)` and `_validate_resume_counters(payload)`.
- Produces: a validated recovery tuple `(epoch, step_in_epoch, global_step, optimizer_step, scheduler_last_epoch, learning_rate, loss, world_size, created_at)` and released GPUs 0-3.

- [ ] **Step 1: Capture the live stall evidence**

Run from the FoCAT repository root:

```bash
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  tmux list-panes -t focat_pretrain -F 'pid=#{pane_pid} command=#{pane_current_command} dead=#{pane_dead}'
  ps -eo pid,ppid,stat,etime,%cpu,%mem,cmd | rg 'torchrun|pretrain.py' | rg -v 'rg '
  nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader
  tail -n 30 logs/focat_loss_monitor.log
} 2>&1 | tee logs/focat_stall_snapshot_20260813.log
```

Expected: four `pretrain.py` workers exist, global step remains 209,449, and no checkpoint newer than step 209,000 exists.

- [ ] **Step 2: Load and validate the latest checkpoint read-only**

```bash
.venv/bin/python - <<'PY'
import json
from ticl.checkpointing import _load_payload, _validate_resume_counters

payload = _load_payload('checkpoints/focat/latest.pt')
_validate_resume_counters(payload)
optimizer_steps = sorted({
    int(state['step'].item() if hasattr(state['step'], 'item') else state['step'])
    for state in payload['optimizer_state']['state'].values()
    if 'step' in state
})
summary = {
    'epoch': payload['epoch'],
    'step_in_epoch': payload['step_in_epoch'],
    'global_step': payload['global_step'],
    'optimizer_steps': optimizer_steps,
    'scheduler_last_epoch': payload['scheduler_state']['last_epoch'],
    'learning_rate': payload['learning_rate'],
    'loss': payload['loss'],
    'world_size': payload['world_size'],
    'rng_state_count': len(payload['rng_states']),
    'created_at': payload['created_at'],
    'source': payload['source'],
}
assert summary['global_step'] == 209000
assert summary['optimizer_steps'] == [209000]
assert summary['world_size'] == summary['rng_state_count'] == 4
print(json.dumps(summary, sort_keys=True))
PY
```

Expected: validation succeeds, loss and learning rate are finite, and model/optimizer/scheduler/RNG metadata are present.

- [ ] **Step 3: Interrupt only the stalled tmux job**

```bash
tmux send-keys -t focat_pretrain:0.0 C-c
```

Poll `tmux has-session -t focat_pretrain` and the four recorded worker PIDs for up to 60 seconds. Do not use a broad `pkill` pattern.

- [ ] **Step 4: Verify shutdown and GPU release**

```bash
ps -eo pid,ppid,stat,etime,cmd | rg 'torchrun|pretrain.py' | rg -v 'rg ' || true
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
```

Expected: the four current worker PIDs are absent and GPUs 0-3 no longer contain FoCAT compute processes. Ignore the historical parent tmux PID if it has no workers or GPU context.

### Task 2: Add a Watchdog-Backed Deterministic Replay Entrypoint

**Files:**
- Create: `diagnose_stall.py`
- Create: `tests/test_diagnose_stall.py`

**Interfaces:**
- Produces: `configure_watchdog(seconds: int) -> None`, which enables all-thread fault dumps and schedules repeated dumps, and `main() -> None`, which executes `pretrain.py` with the original CLI arguments.
- Consumes: environment variable `FOCAT_STACK_DUMP_SECONDS`, defaulting to `60`.

- [ ] **Step 1: Write the failing watchdog tests**

Create `tests/test_diagnose_stall.py`:

```python
from __future__ import annotations

import pytest

import diagnose_stall


def test_configure_watchdog_enables_repeated_all_thread_dumps(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        diagnose_stall.faulthandler,
        "enable",
        lambda *, all_threads: calls.append(("enable", all_threads)),
    )
    monkeypatch.setattr(
        diagnose_stall.faulthandler,
        "dump_traceback_later",
        lambda seconds, *, repeat: calls.append(("later", seconds, repeat)),
    )

    diagnose_stall.configure_watchdog(17)

    assert calls == [("enable", True), ("later", 17, True)]


def test_configure_watchdog_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="positive"):
        diagnose_stall.configure_watchdog(0)
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_diagnose_stall.py
```

Expected: collection fails with `ModuleNotFoundError: No module named 'diagnose_stall'`.

- [ ] **Step 3: Implement the minimal replay wrapper**

Create `diagnose_stall.py`:

```python
from __future__ import annotations

import faulthandler
import os
from pathlib import Path
import runpy
import sys


def configure_watchdog(seconds: int) -> None:
    if seconds < 1:
        raise ValueError("watchdog timeout must be positive")
    faulthandler.enable(all_threads=True)
    faulthandler.dump_traceback_later(seconds, repeat=True)


def main() -> None:
    seconds = int(os.environ.get("FOCAT_STACK_DUMP_SECONDS", "60"))
    configure_watchdog(seconds)
    pretrain_path = Path(__file__).with_name("pretrain.py")
    sys.argv[0] = str(pretrain_path)
    runpy.run_path(str(pretrain_path), run_name="__main__")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run focused and adjacent tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_diagnose_stall.py tests/test_checkpointing.py tests/test_pretrain.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the diagnostic entrypoint**

```bash
git add diagnose_stall.py tests/test_diagnose_stall.py
git commit -m "test: add pretraining stall watchdog"
```

### Task 3: Replay the Exact Stalled RNG Sequence

**Files:**
- Read: `checkpoints/focat/latest.pt`
- Generate: `logs/focat_stall_replay.log`
- Generate only if replay completes: `checkpoints/focat_stall_replay/step_00209450.pt`

**Interfaces:**
- Consumes: the four per-rank RNG states in step 209,000 and the full production config.
- Produces: either `run_stop` at global step 209,450 or repeated Python stack dumps locating the non-returning component.

- [ ] **Step 1: Start a bounded four-rank replay on GPUs 0-3**

```bash
tmux new-session -d -s focat_stall_replay -c /data3/heejin/CausalArena/.worktrees/focat-source-integration/Baselines/CFMs/FoCAT \
"bash -lc 'set -o pipefail; CUDA_VISIBLE_DEVICES=0,1,2,3 FOCAT_STACK_DUMP_SECONDS=60 PYTHONUNBUFFERED=1 .venv/bin/torchrun --standalone --nproc_per_node=4 diagnose_stall.py --profile full --seed 42 --checkpoint-dir checkpoints/focat_stall_replay --checkpoint-every-steps 2000 --keep-last 1 --log-every-steps 1 --resume checkpoints/focat/latest.pt --allow-runtime-change --max-global-steps 209450 2>&1 | tee logs/focat_stall_replay.log'"
```

- [ ] **Step 2: Confirm exact resume and four-GPU placement**

Run:

```bash
tmux ls
nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader
rg '"event": "run_start"' logs/focat_stall_replay.log | tail -n 1
```

Expected: `resumed_global_step` is 209,000, `world_size` is 4, and four new workers occupy physical GPUs 0-3.

- [ ] **Step 3: Apply the replay decision gate**

Poll the replay log without arbitrary sleeping. The outcomes are mutually exclusive:

- Passing outcome: `run_stop` reports global step 209,450 within ten minutes and all losses remain finite.
- Reproduced-stall outcome: global step stops advancing for at least 120 seconds and two consecutive watchdog dumps show the same leaf operation on the same rank or ranks.

For a reproduced stall, capture the last 300 log lines and GPU/process state, interrupt only `focat_stall_replay`, and write a focused root-cause design before any fix. Do not restart production.

- [ ] **Step 4: Validate a passing replay checkpoint**

If the passing outcome occurs, run `_load_payload()` and `_validate_resume_counters()` against `checkpoints/focat_stall_replay/latest.pt` and assert global step 209,450, four RNG states, and optimizer step 209,450. This checkpoint is diagnostic only and must not replace `checkpoints/focat/latest.pt`.

### Task 4: Resume Production with the Approved Cadence

**Files:**
- Append: `logs/focat_pretrain.log`
- Update through normal retention: `checkpoints/focat/latest.pt`, `checkpoints/focat/step_*.pt`

**Interfaces:**
- Consumes: verified production checkpoint at global step 209,000 and a passing deterministic replay.
- Produces: a live `focat_pretrain` tmux session with four workers, finite loss, and `checkpoint_every_steps=2000`.

- [ ] **Step 1: Recreate the production session with one runtime change**

Run only after the replay passes:

```bash
tmux new-session -d -s focat_pretrain -c /data3/heejin/CausalArena/.worktrees/focat-source-integration/Baselines/CFMs/FoCAT \
"bash -lc 'set -o pipefail; CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 .venv/bin/torchrun --standalone --nproc_per_node=4 pretrain.py --profile full --seed 42 --checkpoint-dir checkpoints/focat --checkpoint-every-steps 2000 --keep-last 3 --log-every-steps 1 --resume checkpoints/focat/latest.pt --allow-runtime-change 2>&1 | tee -a logs/focat_pretrain.log'"
```

- [ ] **Step 2: Verify state continuity and healthy loss**

Expected evidence:

- latest `run_start` has `resumed_global_step=209000`, world size 4, seed 42, and source identity recorded;
- the next `train_step` has global step 209001 and finite loss/gradient norm;
- optimizer and scheduler metadata remain those loaded from step 209,000;
- four worker PIDs occupy physical GPUs 0-3;
- no traceback, non-finite loss, NCCL error, or new stall alert appears.

- [ ] **Step 3: Verify the transitional and steady checkpoint intervals**

Observe checkpoint step 210,000, which is an allowed 1,000-step transitional delta because 209,000 is not divisible by 2,000. Then observe checkpoint step 212,000 and assert the steady delta is exactly 2,000. Confirm only the newest three `step_*.pt` files remain and `latest.pt` is a hard link to step 212,000.

- [ ] **Step 4: Run final verification and update the remote branch**

Run:

```bash
.venv/bin/python -m pytest -q tests
git diff --check
git status --short
git push fork codex/focat-full-pretraining
```

Expected: the test suite passes, only runtime artifacts remain untracked, and the existing draft PR includes the cadence design, transition plan, and watchdog entrypoint.
