# FoCAT Distributed Prior Failure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject numerically invalid GP draws within the repository's bounded retry loop, surface future rank-local failures without a 30-minute teardown hang, and resume full pre-training on GPUs 0-3 with checkpoints every 2,000 optimizer steps.

**Architecture:** Extend only `GPPrior.get_batch()`'s existing five-attempt sampling boundary and preserve all valid-draw behavior and prior probabilities. Mark `run_training()` successful only immediately before its normal return, so DDP destruction occurs after success but exceptional workers exit directly to `torchrun`.

**Tech Stack:** Python 3.10, PyTorch 2.7, GPyTorch, pytest, torchrun/DDP, tmux, NVIDIA tooling.

## Global Constraints

- Preserve FoCAT model architecture, MLP/GP prior weights `0.961/0.039`, GP hyperparameter distributions, optimizer, scheduler, seed 42, and global batch size 8.
- Do not turn invalid samples into zero-valued training tasks or catch unrelated errors such as CUDA out-of-memory.
- Keep the original production checkpoint `checkpoints/focat/latest.pt` at global step 209,000 unchanged until a newer production checkpoint is atomically written.
- Require the deterministic four-GPU replay to pass microbatch 2,899 before production resumes.
- Resume production with `--checkpoint-every-steps 2000 --keep-last 3` and append to `logs/focat_pretrain.log`.

---

### Task 1: Add Failing Tests for Invalid GP Draws

**Files:**
- Modify: `tests/test_prior_errors.py`
- Test: `tests/test_prior_errors.py`

**Interfaces:**
- Consumes: `GPPrior.get_batch(batch_size, n_samples, num_features, device)`.
- Produces: regression coverage for a retryable non-finite result and exhausted numerical retries.

- [ ] **Step 1: Add deterministic fake GP components**

Add `FakeGPModel` with `to()` and `__call__()`, plus `FakeLikelihood` whose `sample()` returns two supplied `(batch, samples)` tensors. Keep these test-only classes beside the existing `BrokenGP` patterns.

- [ ] **Step 2: Add the successful second-attempt test**

Monkeypatch `fast_gp.get_model` so attempt one returns two all-NaN tensors and attempt two returns finite zero/one tensors. Call `GPPrior.get_batch()` and assert exactly two model constructions, finite `x/y_0/y_1`, and the expected finite potential outcomes.

- [ ] **Step 3: Add the exhausted retry test**

Monkeypatch every attempt to return all-NaN tensors. Assert five constructions and:

```python
with pytest.raises(FloatingPointError, match="after 5 attempts") as error:
    prior.get_batch(...)
assert isinstance(error.value.__cause__, FloatingPointError)
assert "non-finite" in str(error.value.__cause__)
```

- [ ] **Step 4: Run the focused tests and verify failure**

```bash
.venv/bin/python -m pytest -q tests/test_prior_errors.py
```

Expected: the new tests fail because `GPPrior` currently returns the first NaN result.

### Task 2: Implement Bounded GP Numerical Retries

**Files:**
- Modify: `ticl/priors/fast_gp.py`
- Test: `tests/test_prior_errors.py`

**Interfaces:**
- Produces: `GPPrior.get_batch()` that returns only finite `x`, `sample_0`, and `sample_1`, while retaining the existing five total attempts.

- [ ] **Step 1: Validate each completed draw inside the existing loop**

Immediately after the two samples are drawn, check all returned tensors with `torch.isfinite`. If invalid, assign:

```python
last_error = FloatingPointError("GP prior produced non-finite samples")
```

Print the attempt number, clear CUDA cache only on CUDA, and `continue`. Do not add a broad exception handler.

- [ ] **Step 2: Preserve the terminal error type**

After attempt five, raise `FloatingPointError("GP prior sampling failed after 5 attempts")` when the last error is numerical; otherwise retain the current `RuntimeError` with the same message and cause.

- [ ] **Step 3: Run prior tests**

```bash
.venv/bin/python -m pytest -q tests/test_prior_errors.py
```

Expected: all prior tests pass, including the existing one-attempt CUDA OOM test.

### Task 3: Add Failing Tests for DDP Cleanup Semantics

**Files:**
- Modify: `tests/test_pretrain.py`
- Test: `tests/test_pretrain.py`

**Interfaces:**
- Consumes: `run_training(TrainingOptions)` and `DistributedContext`.
- Produces: proof that successful runs destroy an initialized group exactly once and exceptional runs preserve the original exception without entering DDP destruction.

- [ ] **Step 1: Add a successful cleanup test**

Monkeypatch `_setup_distributed()` to return rank 0, world size 1, CPU, and `initialized_here=True`; monkeypatch `_runtime_source()` to a clean test identity; and monkeypatch `dist.is_initialized()`/`dist.destroy_process_group()`. Run one smoke optimizer step and assert one destroy call.

- [ ] **Step 2: Add an exceptional cleanup test**

Use the same fake context, but monkeypatch `resolve_config()` to raise `RuntimeError("rank-local prior failure")`. Assert `run_training()` raises that exact error and `destroy_process_group()` has no calls.

- [ ] **Step 3: Run focused tests and verify the exceptional case fails**

```bash
.venv/bin/python -m pytest -q tests/test_pretrain.py
```

Expected: the exceptional cleanup test fails because current `finally` destroys the group unconditionally.

### Task 4: Implement Fail-Fast Exceptional DDP Cleanup

**Files:**
- Modify: `pretrain.py`
- Test: `tests/test_pretrain.py`

**Interfaces:**
- Produces: normal `run_training()` return behavior with graceful teardown; exceptional behavior that leaves peer termination to `torchrun` and preserves the original traceback.

- [ ] **Step 1: Track normal completion**

Set `completed_normally = False` before the training `try`. Immediately before returning the final summary, set it to `True`.

- [ ] **Step 2: Gate process-group destruction**

Change the `finally` condition to require `completed_normally`, `context.initialized_here`, and `dist.is_initialized()`. Do not add an `except` block.

- [ ] **Step 3: Run focused and full tests**

```bash
.venv/bin/python -m pytest -q tests/test_pretrain.py
.venv/bin/python -m pytest -q tests
```

Expected: all tests pass.

- [ ] **Step 4: Commit the runtime fix**

```bash
git add ticl/priors/fast_gp.py pretrain.py tests/test_prior_errors.py tests/test_pretrain.py
git commit -m "fix: recover from invalid distributed GP draws"
```

### Task 5: Run Smoke and Deterministic Four-GPU Replay Gates

**Files:**
- Generate: `logs/focat_smoke_gp_retry.log`
- Generate: `logs/focat_ddp_smoke_gp_retry.log`
- Generate: `logs/focat_stall_replay.log`
- Generate: `checkpoints/focat_stall_replay/latest.pt`

**Interfaces:**
- Consumes: fixed source and production step-209,000 checkpoint.
- Produces: verified forward/loss/backward/optimizer/checkpoint/reload behavior and passage beyond global step 209,449.

- [ ] **Step 1: Run bounded single-GPU smoke with resume**

```bash
PYTHONUNBUFFERED=1 .venv/bin/python pretrain.py \
  --profile smoke --device cpu --cpu-threads 1 --seed 4242 \
  --checkpoint-dir checkpoints/focat_smoke_gp_retry \
  --checkpoint-every-steps 1 --keep-last 3 --log-every-steps 1 \
  --max-global-steps 2 2>&1 | tee logs/focat_smoke_gp_retry.log

PYTHONUNBUFFERED=1 .venv/bin/python pretrain.py \
  --profile smoke --device cpu --cpu-threads 1 --seed 4242 \
  --checkpoint-dir checkpoints/focat_smoke_gp_retry \
  --checkpoint-every-steps 1 --keep-last 3 --log-every-steps 1 \
  --resume checkpoints/focat_smoke_gp_retry/latest.pt \
  --max-global-steps 3 2>&1 | tee -a logs/focat_smoke_gp_retry.log
```

Confirm finite loss and restored optimizer/scheduler/global step 3.

- [ ] **Step 2: Run bounded four-GPU smoke**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
  .venv/bin/torchrun --standalone --nproc_per_node=4 pretrain.py \
  --profile smoke --seed 4343 \
  --checkpoint-dir checkpoints/focat_ddp_smoke_gp_retry \
  --checkpoint-every-steps 1 --keep-last 2 --log-every-steps 1 \
  --max-global-steps 2 2>&1 | tee logs/focat_ddp_smoke_gp_retry.log
```

Validate `checkpoints/focat_ddp_smoke_gp_retry/latest.pt` with
`_load_payload()` and `_validate_resume_counters()`; assert global step 2,
optimizer step 2, world size 4, and four RNG states.

- [ ] **Step 3: Replay production through the failed batch**

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 \
  .venv/bin/torchrun --standalone --nproc_per_node=4 pretrain.py \
  --profile full --seed 42 \
  --checkpoint-dir checkpoints/focat_stall_replay \
  --checkpoint-every-steps 2000 --keep-last 1 --log-every-steps 1 \
  --resume checkpoints/focat/latest.pt --allow-runtime-change \
  --max-global-steps 209451 2>&1 | tee logs/focat_stall_replay.log
```

Expected: resume at 209,000, one GP non-finite retry at microbatch 2,899, finite step 209,450 and 209,451 losses, `run_stop`, and diagnostic checkpoint global/optimizer step 209,451. Do not replace the production checkpoint with this replay artifact.

### Task 6: Resume and Verify Production

**Files:**
- Append: `logs/focat_pretrain.log`
- Update through normal retention: `checkpoints/focat/latest.pt`, `checkpoints/focat/step_*.pt`

**Interfaces:**
- Consumes: verified source and production step-209,000 checkpoint.
- Produces: live `focat_pretrain` tmux session on GPUs 0-3 with finite loss and 2,000-step checkpoint cadence.

- [ ] **Step 1: Start production in tmux**

```bash
tmux new-session -d -s focat_pretrain -c /data3/heejin/CausalArena/.worktrees/focat-source-integration/Baselines/CFMs/FoCAT \
"bash -lc 'set -o pipefail; CUDA_VISIBLE_DEVICES=0,1,2,3 PYTHONUNBUFFERED=1 .venv/bin/torchrun --standalone --nproc_per_node=4 pretrain.py --profile full --seed 42 --checkpoint-dir checkpoints/focat --checkpoint-every-steps 2000 --keep-last 3 --log-every-steps 1 --resume checkpoints/focat/latest.pt --allow-runtime-change 2>&1 | tee -a logs/focat_pretrain.log'"
```

- [ ] **Step 2: Verify live training**

Confirm the latest `run_start` resumes global step 209,000, exactly four new worker PIDs occupy physical GPUs 0-3, steps advance with finite loss/gradient norm, and the job passes step 209,451 without a stall or traceback.

- [ ] **Step 3: Verify checkpoint cadence**

Observe checkpoint step 210,000 as the allowed first transition save, then step 212,000. Assert the steady delta is 2,000, exactly three step checkpoints remain, and `latest.pt` is a hard link to the newest one.

- [ ] **Step 4: Final verification and push**

Use `superpowers:verification-before-completion`, rerun `.venv/bin/python -m pytest -q tests`, inspect tmux/GPU/log/checkpoint state, then push `codex/focat-full-pretraining` to `fork` so the existing draft PR contains the audited fix.
