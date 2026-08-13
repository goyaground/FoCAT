# FoCAT Distributed Prior Failure Handling Design

## Goal

Resume the four-GPU full pre-training run past the deterministic failure after
global step 209,449 without changing the FoCAT architecture, valid synthetic
prior distribution, optimizer, scheduler, or checkpoint format. Future
rank-local prior failures must expose their original exception promptly instead
of becoming a 30-minute NCCL timeout.

## Evidence and Root Cause

The production run last logged global step 209,449 at epoch 33,
`step_in_epoch=2898`. Rank 1 then raised `FloatingPointError` while acquiring
microbatch 2,899, before it could enter the scalar loss-validity collective.
Ranks 0, 2, and 3 entered that collective and waited until the 1,800,000 ms NCCL
watchdog expired.

The step-209,000 checkpoint is internally consistent: optimizer step 209,000,
scheduler epoch 32, four RNG states, finite loss and learning rate, and source
commit `fdd7b3f`. Replaying only rank 1's saved Python, NumPy, and Torch CPU RNG
states reproduces the failure at microbatch 2,899 in 34 seconds. The selected
prior is the repository GP prior. Its sampled values are:

- lengthscale: `0.00012053831622086768`
- noise: `0.01`
- outputscale: `0.00003107171648347631`
- sampling: `normal`

Both potential-outcome samples are entirely NaN: 4,608 NaNs in a tensor of
shape `(2304, 2)`. A diagnostic-only monkeypatch that treats this non-finite GP
draw as one failed attempt retries once and advances through microbatch 3,000
with no error. This confirms two distinct causes:

1. `GPPrior.get_batch()` retries factorization exceptions but returns
   numerically non-finite draws without checking them.
2. `run_training()` always calls `destroy_process_group()` while unwinding an
   exception. With peer ranks already in a different collective, that cleanup
   blocks until the NCCL watchdog fires, delaying the original traceback.

## Options Considered

### 1. Bounded GP validation plus fail-fast exceptional cleanup — selected

Validate `x`, `sample_0`, and `sample_1` inside the existing five-attempt GP
loop. A non-finite result consumes one attempt, records a `FloatingPointError`
as its cause, and samples the next repository GP hyperparameters. If all five
attempts are invalid, raise a contextual `FloatingPointError` from the final
cause.

Gracefully destroy the DDP process group only after normal completion. During
exception unwinding, let the failing worker exit with its original traceback;
`torchrun` then terminates its peers instead of the failing worker joining an
unmatched teardown collective. This adds no collective to every training step.

### 2. GP validation only

This is the smallest change and fixes the deterministic microbatch 2,899, but a
future rank-local data or model exception could again be delayed by exceptional
DDP cleanup. It is insufficient for a 1,000-epoch unattended run.

### 3. Convert non-finite potential outcomes to zeros

This would continue immediately but silently inserts degenerate GP tasks and
hides a real numerical failure. It changes the effective training data more
than rejecting a draw that has no finite interpretation, so it is rejected.

## Selected Behavior

In `ticl/priors/fast_gp.py`, keep the current five-attempt loop and existing
factorization-only exception policy. After both GP samples are produced, check
all three returned tensors with `torch.isfinite`. If any tensor is invalid,
print one concise retry message, release cached CUDA memory when applicable,
and continue to the next attempt. Do not retry unrelated runtime errors such as
CUDA out-of-memory.

If no attempt succeeds, use one terminal message for both retryable numerical
classes: `GP prior sampling failed after 5 attempts`. Preserve the last
exception as `__cause__`. Valid GP tensors and their normalization are unchanged.
The prior bag weights remain MLP 0.961 and GP 0.039, and all original GP
hyperparameter distributions remain unchanged; only a numerically undefined
draw is rejected.

In `pretrain.py`, track whether the training body reached its normal return.
The `finally` block calls `dist.destroy_process_group()` only for that normal
path. It does not catch, replace, or suppress `Exception`, `KeyboardInterrupt`,
or `SystemExit`; the original failure reaches `torchrun` immediately.

## Tests and Production Gate

Add focused tests that prove:

- a first GP draw containing NaN is retried and a second finite draw is
  returned;
- five non-finite GP draws raise the contextual error with the numerical error
  retained as the cause;
- unrelated runtime errors are still attempted exactly once;
- normal DDP cleanup destroys an initialized process group;
- exceptional DDP cleanup does not enter process-group destruction and does
  not suppress the supplied exception.

Run the complete `tests/` suite and the existing single-GPU and four-GPU smoke
pipelines. Then replay production from step 209,000 through at least global
step 209,451 on GPUs 0-3. The replay must log the GP retry, advance beyond the
previous failure, keep all losses finite, and save/reload its bounded checkpoint.
Only after that gate passes may production resume from the original step-209,000
checkpoint with `--checkpoint-every-steps 2000 --keep-last 3`.
