# FoCAT Checkpoint Cadence Design

## Goal

Reduce checkpoint write overhead during the live four-GPU FoCAT full
pre-training run without changing its model, synthetic prior, optimizer,
scheduler, training counters, or resume guarantees.

## Decision

Change only the live command-line checkpoint interval from 1,000 to 2,000
optimizer steps and retain the newest three step checkpoints. Continue using
`checkpoints/focat/latest.pt` as the atomic hard link to the newest retained
checkpoint.

At the current gradient accumulation of two, an epoch contains 4,000 optimizer
steps. The new interval therefore saves twice per epoch, approximately every
12 minutes at the observed throughput. Adaptive accumulation changes this to
about once per epoch after epoch 50 and once every two epochs after epoch 200.

The alternatives were:

- keep 1,000 steps: strongest recovery granularity but writes a 2.5 GB state
  about every six minutes;
- use 4,000 steps: one checkpoint per current epoch, but grows to roughly a
  90-minute recovery window after the later accumulation increase;
- use 2,000 steps: the selected balance, keeping the expected recovery window
  between about 12 and 45 minutes across the planned accumulation stages.

## Safe Transition

1. Wait for the next production `checkpoint_saved` event so the restart begins
   from a fresh, complete checkpoint.
2. Record its epoch, microbatch, global step, optimizer step, scheduler state,
   learning rate, source identity, and creation time.
3. Send an interrupt only to the `focat_pretrain` tmux session and verify all
   four FoCAT workers have exited and GPUs 0-3 are released.
4. Recreate `focat_pretrain` with the identical command except for
   `--checkpoint-every-steps 2000`, resuming from
   `checkpoints/focat/latest.pt` and appending to the existing log.
5. Verify the resumed global step, finite loss, learning rate, four worker PIDs,
   GPU 0-3 activity, and absence of a new traceback or non-finite error.
6. Observe the next checkpoint and verify that its global-step delta from the
   preceding checkpoint follows the new 2,000-step cadence. A shorter first
   delta is allowed only when the resume checkpoint was not divisible by 2,000;
   the following delta must be exactly 2,000.

## Scope and Failure Handling

No source-code or checkpoint-format change is required. Existing checkpoints
are not deleted manually; normal `keep_last=3` retention handles them after the
next saves. If the fresh checkpoint cannot be loaded consistently, the current
training process is left running. If the resumed process fails, the last
verified checkpoint remains intact and the failure is reported rather than
restarting repeatedly.
