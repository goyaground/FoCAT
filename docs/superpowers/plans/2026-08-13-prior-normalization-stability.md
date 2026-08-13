# FoCAT Prior Normalization Stability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent finite extreme outcomes from the original FoCAT MLP prior from overflowing during CATE normalization, then safely resume the existing four-GPU full pre-training run.

**Architecture:** Keep the original MLP/GP prior sampling and RNG sequence unchanged. Stabilize the existing separate-arm centering and shared-scale normalization by dividing both arms by a common per-task magnitude before computing statistics, and extend the MLP's existing invalid-output fallback from NaN-only to all non-finite values. Preserve the training loop's fail-fast numeric checks as the final guard.

**Tech Stack:** Python 3.10, PyTorch, pytest, torchrun/DDP, tmux

## Global Constraints

- Preserve the original FoCAT Transformer, hypernetwork, generated MLP, histogram loss, MLP/GP prior distributions, prior weights, and treatment assignment mechanism.
- Do not discard or resample an extreme task; keep the RNG sequence unchanged.
- Limit code changes to synthetic-prior numerical safety and focused regression tests.
- Any non-finite value that escapes the prior adapter remains a hard training error.
- Resume from `checkpoints/focat/latest.pt`; do not search for another checkpoint.

---

## File Structure

- Modify `ticl/priors/classification_adapter.py`: stable CATE normalization and an explicit finite-input contract.
- Modify `ticl/priors/mlp.py`: make the existing invalid-output fallback cover NaN and infinity.
- Modify `tests/test_prior_errors.py`: focused, deterministic regression tests for both failure modes.
- No model, loss, configuration, or checkpoint format file changes.

### Task 1: Stabilize CATE normalization

**Files:**
- Modify: `ticl/priors/classification_adapter.py:105-111`
- Test: `tests/test_prior_errors.py`

**Interfaces:**
- Consumes: `ClassificationAdapter.normalize_cate(self, y_0: torch.Tensor, y_1: torch.Tensor)` with tensors shaped `(n_samples, batch_size)`.
- Produces: two finite tensors of the same shape and dtype, using separate arm means and one common per-task standard deviation.

- [ ] **Step 1: Write failing extreme-value and non-finite-input tests**

Add the import and tests below to `tests/test_prior_errors.py`:

```python
from ticl.priors.classification_adapter import ClassificationAdapter


def test_cate_normalization_is_stable_near_float32_limit() -> None:
    adapter = ClassificationAdapter.__new__(ClassificationAdapter)
    y_0 = torch.tensor(
        [[-1.7e38], [-1.2e38], [-7.0e37], [-2.0e37]],
        dtype=torch.float32,
    )
    y_1 = torch.tensor(
        [[-1.6e38], [-1.1e38], [-6.0e37], [-1.0e37]],
        dtype=torch.float32,
    )
    combined = torch.cat((y_0.double(), y_1.double()), dim=0)
    sigma = combined.std(dim=0, keepdim=True)
    expected_0 = ((y_0.double() - y_0.double().mean(0, keepdim=True)) / sigma).float()
    expected_1 = ((y_1.double() - y_1.double().mean(0, keepdim=True)) / sigma).float()

    actual_0, actual_1 = adapter.normalize_cate(y_0, y_1)

    assert torch.isfinite(actual_0).all()
    assert torch.isfinite(actual_1).all()
    torch.testing.assert_close(actual_0, expected_0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_1, expected_1, rtol=1e-5, atol=1e-6)


def test_cate_normalization_rejects_nonfinite_inputs() -> None:
    adapter = ClassificationAdapter.__new__(ClassificationAdapter)
    y_0 = torch.tensor([[0.0], [float("inf")]])
    y_1 = torch.tensor([[1.0], [2.0]])

    with pytest.raises(FloatingPointError, match="potential outcomes must be finite"):
        adapter.normalize_cate(y_0, y_1)
```

- [ ] **Step 2: Run the tests and verify the current implementation fails**

Run:

```bash
.venv/bin/python -m pytest -q \
  tests/test_prior_errors.py::test_cate_normalization_is_stable_near_float32_limit \
  tests/test_prior_errors.py::test_cate_normalization_rejects_nonfinite_inputs
```

Expected: the extreme-value test observes infinity or a mismatch, and the explicit-infinity test does not raise the required error.

- [ ] **Step 3: Implement scale-first normalization**

Replace `normalize_cate()` with:

```python
def normalize_cate(self, y_0, y_1):
    combined = torch.cat((y_0, y_1), dim=0)
    if not torch.isfinite(combined).all():
        raise FloatingPointError(
            "potential outcomes must be finite before CATE normalization"
        )
    scale = combined.abs().amax(dim=0, keepdim=True)
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)
    scaled_0 = y_0 / scale
    scaled_1 = y_1 / scale
    sigma = torch.std(torch.cat((scaled_0, scaled_1), dim=0), dim=0, keepdim=True)
    sigma[sigma < 1e-6] = 1
    mean_0 = torch.mean(scaled_0, dim=0, keepdim=True)
    mean_1 = torch.mean(scaled_1, dim=0, keepdim=True)
    return (scaled_0 - mean_0) / sigma, (scaled_1 - mean_1) / sigma
```

- [ ] **Step 4: Run the focused tests and the prior test module**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_prior_errors.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the independently working normalization fix**

```bash
git add ticl/priors/classification_adapter.py tests/test_prior_errors.py
git commit -m "fix: stabilize FoCAT prior normalization"
```

### Task 2: Complete the MLP prior's existing non-finite fallback

**Files:**
- Modify: `ticl/priors/mlp.py:180-191`
- Test: `tests/test_prior_errors.py`

**Interfaces:**
- Consumes: `MLP.forward()` outputs `x`, `y_0`, and `y_1`.
- Produces: the existing all-zero fallback when any of those tensors contains NaN, positive infinity, or negative infinity.

- [ ] **Step 1: Write a failing infinity regression test**

Add to `tests/test_prior_errors.py`:

```python
from torch import nn
from ticl.priors.mlp import MLP


class InfiniteOutput(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.full((*value.shape[:-1], 1), float("inf"), device=value.device)


class ZeroOutput(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return torch.zeros((*value.shape[:-1], 1), device=value.device)


def test_mlp_existing_fallback_handles_infinity() -> None:
    model = MLP(
        "cpu", 2, 1, 4, "normal",
        num_layers=2,
        prior_mlp_hidden_dim=4,
        prior_mlp_activations=nn.ReLU,
        noise_std=0.0,
        y_is_effect=False,
        pre_sample_weights=False,
        prior_mlp_dropout_prob=0.0,
        pre_sample_causes=False,
        prior_mlp_scale_weights_sqrt=True,
        random_feature_rotation=False,
        add_uninformative_features=False,
        is_causal=False,
        num_causes=2,
        block_wise_dropout=False,
        init_std=1.0,
        sort_features=False,
        in_clique=False,
    )
    model.layers_0 = nn.Sequential(nn.Identity(), InfiniteOutput())
    model.layers_eff = nn.Sequential(ZeroOutput())

    x, y_0, y_1 = model()

    assert torch.count_nonzero(x) == 0
    assert torch.count_nonzero(y_0) == 0
    assert torch.count_nonzero(y_1) == 0
```

- [ ] **Step 2: Run the test and verify it fails before the code change**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_prior_errors.py::test_mlp_existing_fallback_handles_infinity
```

Expected: the outcome tensors still contain infinity.

- [ ] **Step 3: Extend the existing check without changing fallback semantics**

In `MLP.forward()`, replace the three NaN-only conditions with `not torch.isfinite(...).all()`, update the diagnostic counts to `(~torch.isfinite(...)).sum()`, and keep the existing assignments of zero to `x`, `y_0`, and `y_1` unchanged.

- [ ] **Step 4: Run the focused and full prior tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/test_prior_errors.py
```

Expected: all tests pass.

- [ ] **Step 5: Commit the independently working fallback fix**

```bash
git add ticl/priors/mlp.py tests/test_prior_errors.py
git commit -m "fix: handle infinite MLP prior outputs"
```

### Task 3: Verify the exact failed batch and repository test suite

**Files:**
- No production-file changes.
- Read: `checkpoints/focat/latest.pt`

**Interfaces:**
- Consumes: rank 2's Python, NumPy, and CPU Torch RNG states from the checkpoint saved at epoch 25, batch 4,000.
- Produces: deterministic evidence that replayed batch 5,710 has finite `x`, observed `y`, treatment, and CATE targets.

- [ ] **Step 1: Run all unit and smoke tests**

```bash
.venv/bin/python -m pytest -q tests
```

Expected: all tests pass.

- [ ] **Step 2: Replay the rank 2 prior stream through the failed batch**

Run this from the FoCAT repository root:

```bash
.venv/bin/python - <<'PY'
import random
import numpy as np
import torch
from ticl.dataloader import get_dataloader

payload = torch.load(
    "checkpoints/focat/latest.pt",
    map_location="cpu",
    weights_only=False,
    mmap=True,
)
loader = get_dataloader(
    payload["config"]["prior"],
    payload["config"]["dataloader"],
    device="cpu",
)
state = payload["rng_states"][2]
random.setstate(state["python"])
np.random.set_state(state["numpy"])
torch.set_rng_state(state["torch"])
loader.epoch_count = payload["epoch"] - 1
iterator = iter(loader)
for offset in range(1, 1711):
    data, targets, single_eval_pos = next(iterator)
x, observed_y, treatment = data
for name, value in {
    "x": x,
    "observed_y": observed_y,
    "treatment": treatment,
    "targets": targets,
}.items():
    assert torch.isfinite(value).all(), name
print({
    "batch": payload["step_in_epoch"] + offset,
    "single_eval_pos": single_eval_pos,
    "target_shape": tuple(targets.shape),
    "target_min": float(targets.min()),
    "target_max": float(targets.max()),
})
PY
```

Expected: batch `5710`, target shape `(1152, 2)`, and finite extrema.

- [ ] **Step 3: Confirm only intended tracked files changed**

```bash
git status --short --untracked-files=no
git log -4 --oneline
```

Expected: no uncommitted tracked changes; the design, plan, normalization, and fallback commits are visible.

### Task 4: Verify four-GPU resume continuity with a bounded DDP run

**Files:**
- Read: `checkpoints/focat/latest.pt`
- Create runtime artifact: `checkpoints/focat_resume_smoke/latest.pt`
- Create runtime artifact: `logs/focat_resume_smoke.log`

**Interfaces:**
- Consumes: the original four-rank checkpoint at global step 178,000.
- Produces: a reloaded four-rank checkpoint at global step 178,002 with continuous optimizer and scheduler state.

- [ ] **Step 1: Confirm GPUs 0-3 are free and record the baseline metadata**

```bash
nvidia-smi
.venv/bin/python - <<'PY'
import torch
p = torch.load("checkpoints/focat/latest.pt", map_location="cpu", weights_only=False, mmap=True)
print({k: p[k] for k in ("epoch", "step_in_epoch", "global_step", "learning_rate", "aggregate_k_gradients")})
print("scheduler_last_epoch", p["scheduler_state"]["last_epoch"])
PY
```

Expected: GPUs 0-3 have no FoCAT process and the checkpoint reports epoch 25, batch 4,000, and global step 178,000.

- [ ] **Step 2: Run two bounded optimizer steps on four GPUs**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
  .venv/bin/torchrun --standalone --nproc_per_node=4 pretrain.py \
  --profile full --seed 42 \
  --checkpoint-dir checkpoints/focat_resume_smoke \
  --checkpoint-every-steps 2 --keep-last 1 --log-every-steps 1 \
  --max-global-steps 178002 \
  --resume checkpoints/focat/latest.pt --allow-runtime-change \
  2>&1 | tee logs/focat_resume_smoke.log
```

Expected: two finite loss records, no distributed error, and a final checkpoint.

- [ ] **Step 3: Verify saved resume continuity**

```bash
.venv/bin/python - <<'PY'
import torch
p = torch.load("checkpoints/focat_resume_smoke/latest.pt", map_location="cpu", weights_only=False, mmap=True)
steps = {
    int(s["step"].item() if hasattr(s["step"], "item") else s["step"])
    for s in p["optimizer_state"]["state"].values()
}
print({
    "epoch": p["epoch"],
    "step_in_epoch": p["step_in_epoch"],
    "global_step": p["global_step"],
    "learning_rate": p["learning_rate"],
    "optimizer_steps": sorted(steps),
    "scheduler_last_epoch": p["scheduler_state"]["last_epoch"],
})
assert p["global_step"] == 178002
assert steps == {178002}
assert p["scheduler_state"]["last_epoch"] == 24
PY
```

Expected: global and optimizer steps both equal 178,002, with scheduler epoch 24 and the resumed learning rate unchanged.

### Task 5: Resume and observe full pre-training beyond the failed point

**Files:**
- Append runtime log: `logs/focat_pretrain.log`
- Update runtime checkpoints: `checkpoints/focat/latest.pt`, `checkpoints/focat/step_*.pt`

**Interfaces:**
- Consumes: the production checkpoint at global step 178,000 and GPUs 0-3.
- Produces: a live `focat_pretrain` tmux session whose global step exceeds 178,855 and continues training.

- [ ] **Step 1: Start the production resume in tmux**

```bash
tmux new-session -d -s focat_pretrain \
  "bash -lc 'cd /data3/heejin/CausalArena/.worktrees/focat-source-integration/Baselines/CFMs/FoCAT && CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 .venv/bin/torchrun --standalone --nproc_per_node=4 pretrain.py --profile full --seed 42 --checkpoint-dir checkpoints/focat --checkpoint-every-steps 1000 --keep-last 3 --log-every-steps 1 --resume checkpoints/focat/latest.pt --allow-runtime-change 2>&1 | tee -a logs/focat_pretrain.log'"
```

Expected: the session starts without replacing any existing live session.

- [ ] **Step 2: Confirm all four ranks and GPUs are active**

```bash
tmux ls
nvidia-smi
pgrep -af "torchrun.*pretrain.py|pretrain.py --profile full"
```

Expected: `focat_pretrain` exists and one worker is active on each of GPUs 0, 1, 2, and 3.

- [ ] **Step 3: Observe the run beyond the formerly failing optimizer step**

```bash
tail -n 50 logs/focat_pretrain.log
```

Repeat until a finite `train_step` record has `global_step` greater than 178,855. Confirm there is no `FloatingPointError`, `ChildFailedError`, traceback, NaN loss, or non-finite gradient after the resumed `run_start` record.

- [ ] **Step 4: Confirm checkpoint and monitoring state**

```bash
.venv/bin/python - <<'PY'
import torch
p = torch.load("checkpoints/focat/latest.pt", map_location="cpu", weights_only=False, mmap=True)
print({k: p[k] for k in ("epoch", "step_in_epoch", "global_step", "loss", "learning_rate", "created_at")})
PY
tmux ls
nvidia-smi
```

Expected: training remains live; the latest checkpoint advances at the next 1,000-step boundary and contains finite loss plus continuous optimizer/scheduler state.
