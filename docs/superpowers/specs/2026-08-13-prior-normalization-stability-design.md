# FoCAT Prior Normalization Stability Design

## Context

The four-GPU full pre-training run stopped at epoch 25, microbatch 5,710 on
rank 2. Replaying that batch from the rank-specific RNG state in
`checkpoints/focat/latest.pt` reproduced the failure exactly. One task sampled
the repository's MLP prior with 16 layers and `init_std` about 10.47. Its raw
potential outcomes remained finite but reached roughly `-1.7e38`.

`ClassificationAdapter.normalize_cate()` computed arm means and the shared
standard deviation directly in float32. The mean overflowed to infinity, so
finite raw outcomes became infinite normalized outcomes and `y1 - y0` became
NaN. The Transformer, histogram loss, optimizer, and saved checkpoint were not
the first source of the non-finite value.

## Decision

Keep every sampled synthetic task and make the existing CATE normalization
numerically stable by scaling each task before computing its statistics.

For each batch column:

1. Compute the largest absolute value across both potential-outcome arms.
2. Replace only a zero scale with one.
3. Divide both arms by that common scale.
4. Compute the two arm means and their shared standard deviation on the scaled
   values.
5. Apply the same separate-arm centering and shared-standard-deviation
   normalization already used by FoCAT.

For finite inputs this is algebraically equivalent to the existing
normalization. It avoids overflow without changing the MLP/GP mixture,
hyperparameter distributions, treatment assignment, model architecture, or
RNG sequence. The task will not be discarded or resampled.

The MLP prior's existing NaN fallback will also recognize positive and negative
infinity. This retains its current zero-output fallback semantics while closing
the incomplete non-finite check. If a non-finite value nevertheless escapes
the adapter, generation will fail immediately with a specific error instead of
letting the loss hide one task through `nanmean`.

## Scope

The change is limited to synthetic-prior numerical safety:

- stable normalization in `ClassificationAdapter.normalize_cate()`;
- complete finite-value checking in the MLP prior and adapter;
- focused regression tests for extreme finite values and explicit non-finite
  inputs.

It does not change the FoCAT Transformer, hypernetwork, generated MLP,
histogram loss, prior distributions, prior weights, or treatment mechanism.
It does not add retries or alter the random sampling sequence.

## Verification

Tests will be added before implementation and must demonstrate the current
failure first. Verification then proceeds in increasing cost order:

1. Unit tests confirm that values near the float32 limit normalize to finite
   outputs with approximately zero arm means and match a float64 reference.
2. Unit tests confirm that the existing MLP fallback handles all non-finite
   values, not only NaN.
3. Rank 2's saved RNG state is replayed through microbatch 5,710 and all prior
   tensors are checked for finiteness.
4. The checkpoint is resumed for a short multi-GPU save/reload smoke run and
   optimizer step, scheduler state, learning rate, and global step continuity
   are checked.
5. The full four-GPU run is resumed in `focat_pretrain`, observed beyond the
   formerly failing optimizer step, and its loss/checkpoint logs are checked.

Any remaining non-finite prior output is a hard error with tensor and task
context. No broad exception or silent resampling is introduced.
