from __future__ import annotations

import pytest
import torch

from ticl.dataloader import PriorDataLoader
from ticl.priors import fast_gp


class AssertionPrior:
    def get_batch(self, **_kwargs):
        raise AssertionError("synthetic prior shape failure")


def test_dataloader_does_not_hide_prior_assertions() -> None:
    loader = PriorDataLoader(
        prior=AssertionPrior(),
        num_steps=1,
        batch_size=1,
        min_eval_pos=2,
        n_samples=4,
        device="cpu",
        num_features=2,
        treatment_part={"distribution": "uniform", "min": 0.1, "max": 0.5},
    )

    with pytest.raises(AssertionError, match="synthetic prior shape failure"):
        next(iter(loader))


def test_gp_prior_retries_only_five_times_then_preserves_cause(monkeypatch) -> None:
    attempts = []

    class BrokenGP:
        def to(self, _device):
            return self

        def __call__(self, _x):
            attempts.append(1)
            raise RuntimeError("factorization failed")

    monkeypatch.setattr(
        fast_gp,
        "get_model",
        lambda _x, _y, _hyperparameters: (BrokenGP(), lambda value: value),
    )
    prior = fast_gp.GPPrior(
        {
            "outputscale": 1.0,
            "lengthscale": 1.0,
            "noise": 0.01,
            "sampling": "normal",
        }
    )

    with pytest.raises(RuntimeError, match="after 5 attempts") as error:
        prior.get_batch(
            batch_size=1,
            n_samples=4,
            num_features=2,
            device="cpu",
        )

    assert len(attempts) == 5
    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "factorization failed"


def test_gp_prior_does_not_retry_unrelated_runtime_errors(monkeypatch) -> None:
    attempts = []

    class BrokenGP:
        def to(self, _device):
            return self

        def __call__(self, _x):
            attempts.append(1)
            raise RuntimeError("CUDA out of memory")

    monkeypatch.setattr(
        fast_gp,
        "get_model",
        lambda _x, _y, _hyperparameters: (BrokenGP(), lambda value: value),
    )
    prior = fast_gp.GPPrior(
        {
            "outputscale": 1.0,
            "lengthscale": 1.0,
            "noise": 0.01,
            "sampling": "normal",
        }
    )

    with pytest.raises(RuntimeError, match="CUDA out of memory"):
        prior.get_batch(
            batch_size=1,
            n_samples=4,
            num_features=2,
            device="cpu",
        )

    assert len(attempts) == 1
