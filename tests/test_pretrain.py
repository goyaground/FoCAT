from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

import pretrain
from pretrain import (
    TrainingOptions,
    _validate_batch_numerics,
    _validate_resume_identity,
    resolve_config,
    run_training,
)
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


def test_partial_nan_share_is_a_fatal_training_error() -> None:
    with pytest.raises(FloatingPointError, match="non-finite values"):
        _validate_batch_numerics(
            torch.tensor(1.25),
            torch.tensor(0.25),
            epoch=3,
            batch_index=17,
        )


def test_resume_identity_rejects_config_and_source_changes() -> None:
    current_config = resolve_config("smoke", world_size=1)
    current_source = {
        "upstream_url": "https://github.com/NTAILab/FoCAT.git",
        "upstream_base_commit": "608006595eaf2eda40c36f097c369d1efd65d500",
        "runtime_commit": "current-commit",
        "runtime_dirty": False,
    }
    payload = {
        "config": copy.deepcopy(current_config),
        "source": copy.deepcopy(current_source),
    }

    _validate_resume_identity(payload, current_config, current_source)

    changed_config = copy.deepcopy(payload)
    changed_config["config"]["prior"]["num_features"] = 99
    with pytest.raises(ValueError, match="training config"):
        _validate_resume_identity(changed_config, current_config, current_source)

    changed_source = copy.deepcopy(payload)
    changed_source["source"]["runtime_commit"] = "different-commit"
    with pytest.raises(ValueError, match="runtime source"):
        _validate_resume_identity(changed_source, current_config, current_source)

    _validate_resume_identity(
        changed_source,
        current_config,
        current_source,
        allow_runtime_change=True,
    )


def test_smoke_training_resumes_optimizer_scheduler_and_global_step(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        pretrain,
        "_runtime_source",
        lambda: {
            "upstream_url": "https://github.com/NTAILab/FoCAT.git",
            "upstream_base_commit": "608006595eaf2eda40c36f097c369d1efd65d500",
            "runtime_commit": "test-runtime",
            "runtime_dirty": False,
        },
    )
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


def test_successful_distributed_training_destroys_process_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = pretrain.DistributedContext(
        rank=0,
        local_rank=0,
        world_size=1,
        device=torch.device("cpu"),
        initialized_here=True,
    )
    destroyed = []
    monkeypatch.setattr(pretrain, "_setup_distributed", lambda _device: context)
    monkeypatch.setattr(
        pretrain,
        "_runtime_source",
        lambda: {
            "upstream_url": "https://github.com/NTAILab/FoCAT.git",
            "upstream_base_commit": "608006595eaf2eda40c36f097c369d1efd65d500",
            "runtime_commit": "test-runtime",
            "runtime_dirty": False,
        },
    )
    monkeypatch.setattr(pretrain.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        pretrain.dist,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )

    summary = run_training(
        TrainingOptions(
            profile="smoke",
            checkpoint_dir=tmp_path / "checkpoints",
            seed=456,
            max_global_steps=1,
            checkpoint_every_steps=1,
            keep_last=1,
            log_every_steps=1,
            device="cpu",
            cpu_threads=1,
        )
    )

    assert summary["global_step"] == 1
    assert destroyed == [True]


def test_rank_local_failure_does_not_enter_collective_cleanup(
    monkeypatch,
) -> None:
    context = pretrain.DistributedContext(
        rank=1,
        local_rank=1,
        world_size=4,
        device=torch.device("cpu"),
        initialized_here=True,
    )
    destroyed = []
    monkeypatch.setattr(pretrain, "_setup_distributed", lambda _device: context)
    monkeypatch.setattr(
        pretrain,
        "_runtime_source",
        lambda: {
            "upstream_url": "https://github.com/NTAILab/FoCAT.git",
            "upstream_base_commit": "608006595eaf2eda40c36f097c369d1efd65d500",
            "runtime_commit": "test-runtime",
            "runtime_dirty": False,
        },
    )
    monkeypatch.setattr(
        pretrain,
        "resolve_config",
        lambda _profile, _world_size: (_ for _ in ()).throw(
            RuntimeError("rank-local prior failure")
        ),
    )
    monkeypatch.setattr(pretrain.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        pretrain.dist,
        "destroy_process_group",
        lambda: destroyed.append(True),
    )

    with pytest.raises(RuntimeError, match="rank-local prior failure"):
        run_training(TrainingOptions(profile="smoke", device="cpu"))

    assert destroyed == []
