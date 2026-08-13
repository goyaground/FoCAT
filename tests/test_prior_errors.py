from __future__ import annotations

import pytest
import torch

from ticl.dataloader import PriorDataLoader
from ticl.priors import fast_gp
from ticl.priors.classification_adapter import ClassificationAdapter


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
    expected_0 = (
        (y_0.double() - y_0.double().mean(0, keepdim=True)) / sigma
    ).float()
    expected_1 = (
        (y_1.double() - y_1.double().mean(0, keepdim=True)) / sigma
    ).float()

    actual_0, actual_1 = adapter.normalize_cate(y_0, y_1)

    assert torch.isfinite(actual_0).all()
    assert torch.isfinite(actual_1).all()
    torch.testing.assert_close(actual_0, expected_0, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual_1, expected_1, rtol=1e-5, atol=1e-6)


def test_cate_normalization_rejects_nonfinite_inputs() -> None:
    adapter = ClassificationAdapter.__new__(ClassificationAdapter)
    y_0 = torch.tensor([[0.0], [float("inf")]])
    y_1 = torch.tensor([[1.0], [2.0]])

    with pytest.raises(
        FloatingPointError,
        match="potential outcomes must be finite",
    ):
        adapter.normalize_cate(y_0, y_1)
