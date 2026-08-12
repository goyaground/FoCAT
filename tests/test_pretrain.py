from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pretrain import TrainingOptions, resolve_config, run_training
from ticl.dataloader import get_dataloader


def optimizer_steps(checkpoint: dict) -> set[int]:
    return {
        int(state["step"].item() if hasattr(state["step"], "item") else state["step"])
        for state in checkpoint["optimizer_state"]["state"].values()
    }


def test_full_config_preserves_repository_model_prior_and_global_batch() -> None:
    config = resolve_config("full", world_size=4)

    assert config["transformer"]["emsize"] == 512
    assert config["transformer"]["nlayers"] == 12
    assert config["transformer"]["nhead"] == 8
    assert config["transformer"]["classification_task"] is False
    assert config["transformer"]["y_encoder"] == "linear"
    assert config["mothernet"]["predicted_hidden_layer_size"] == 512
    assert config["mothernet"]["predicted_hidden_layers"] == 2
    assert config["mothernet"]["weight_embedding_rank"] == 16
    assert config["mothernet"]["bins_output_n"] == 2048
    assert config["prior"]["num_features"] == 100
    assert config["prior"]["n_samples"] == 1152
    assert config["prior"]["prior_type"] == "prior_bag"
    assert config["dataloader"]["treatment_part"] == {
        "distribution": "uniform",
        "min": 0.1,
        "max": 0.5,
    }
    assert config["dataloader"]["batch_size"] == 2
    assert config["runtime"]["global_batch_size"] == 8
    assert config["dataloader"]["num_steps"] == 8000
    assert config["optimizer"]["epochs"] == 1000
    assert config["optimizer"]["learning_rate"] == 2e-5
    assert config["optimizer"]["warmup_epochs"] == 20
    assert config["optimizer"]["learning_rate_schedule"] == "cosine"
    loader = get_dataloader(
        config["prior"], config["dataloader"], device="cpu"
    )
    assert loader.prior.prior_weights == {"mlp": 0.961, "gp": 0.039}


def test_full_config_rejects_world_size_that_changes_effective_batch() -> None:
    with pytest.raises(ValueError, match="global batch size 8"):
        resolve_config("full", world_size=3)


def test_smoke_config_reduces_only_scale_and_keeps_training_semantics() -> None:
    config = resolve_config("smoke", world_size=1)

    assert config["prior"]["prior_type"] == "prior_bag"
    assert config["dataloader"]["treatment_part"] == {
        "distribution": "uniform",
        "min": 0.1,
        "max": 0.5,
    }
    assert config["mothernet"]["bins_output_n"] == 2048
    assert config["mothernet"]["bins_to_sum_n"] == 10
    assert config["optimizer"]["learning_rate_schedule"] == "cosine"
    assert config["optimizer"]["weight_decay"] == 0.0
    assert config["dataloader"]["batch_size"] == 1
    assert config["dataloader"]["num_steps"] == 2
    assert config["optimizer"]["epochs"] == 2


def test_smoke_training_resumes_optimizer_scheduler_and_global_step(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "checkpoints"
    first = run_training(
        TrainingOptions(
            profile="smoke",
            checkpoint_dir=checkpoint_dir,
            seed=123,
            max_global_steps=2,
            checkpoint_every_steps=1,
            keep_last=3,
            log_every_steps=1,
            device="cpu",
            cpu_threads=1,
        )
    )
    first_checkpoint = torch.load(
        checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False
    )

    assert first["global_step"] == 2
    assert first["epoch"] == 1
    assert first["step_in_epoch"] == 2
    assert torch.isfinite(torch.tensor(first["loss"]))
    assert optimizer_steps(first_checkpoint) == {2}
    first_scheduler_epoch = first_checkpoint["scheduler_state"]["last_epoch"]

    resumed = run_training(
        TrainingOptions(
            profile="smoke",
            checkpoint_dir=checkpoint_dir,
            resume=checkpoint_dir / "latest.pt",
            seed=123,
            max_global_steps=3,
            checkpoint_every_steps=1,
            keep_last=3,
            log_every_steps=1,
            device="cpu",
            cpu_threads=1,
        )
    )
    resumed_checkpoint = torch.load(
        checkpoint_dir / "latest.pt", map_location="cpu", weights_only=False
    )

    assert resumed["global_step"] == 3
    assert resumed["epoch"] == 2
    assert resumed["step_in_epoch"] == 1
    assert torch.isfinite(torch.tensor(resumed["loss"]))
    assert optimizer_steps(resumed_checkpoint) == {3}
    assert resumed_checkpoint["scheduler_state"]["last_epoch"] == first_scheduler_epoch
