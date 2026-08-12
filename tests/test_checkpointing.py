from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pytest
import torch

from ticl.checkpointing import (
    capture_rng_state,
    checkpoint_payload,
    load_training_checkpoint,
    restore_rng_state,
    save_training_checkpoint,
)
from ticl.model_builder import get_model, load_model
from ticl.model_configs import get_model_default_config


def make_optimizer_state(steps: int = 3):
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=4, eta_min=1e-5
    )
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        [
            torch.optim.lr_scheduler.LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=1
            ),
            base_scheduler,
        ],
        milestones=[1],
    )
    for _ in range(steps):
        optimizer.zero_grad()
        model(torch.ones(2, 2)).sum().backward()
        optimizer.step()
    scheduler.step()
    return model, optimizer, scheduler


def optimizer_steps(optimizer: torch.optim.Optimizer) -> set[int]:
    return {
        int(state["step"].item() if hasattr(state["step"], "item") else state["step"])
        for state in optimizer.state.values()
    }


def make_payload(global_step: int = 3):
    model, optimizer, scheduler = make_optimizer_state(global_step)
    state = capture_rng_state()
    payload = checkpoint_payload(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=2,
        step_in_epoch=7,
        global_step=global_step,
        config={"model_type": "test", "nested": {"value": 1}},
        seed=42,
        world_size=1,
        rng_states=[state],
        loss=1.25,
        learning_rate=optimizer.param_groups[0]["lr"],
        aggregate_k_gradients=1,
        source={"upstream_commit": "a" * 40},
        started_at="2026-08-12T00:00:00+00:00",
    )
    return payload, model, optimizer, scheduler


def test_rng_state_round_trip() -> None:
    random.seed(19)
    np.random.seed(19)
    torch.manual_seed(19)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(3))

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    restore_rng_state(state)
    actual = (random.random(), np.random.rand(), torch.rand(3))

    assert actual[0] == expected[0]
    assert actual[1] == expected[1]
    torch.testing.assert_close(actual[2], expected[2])


def test_checkpoint_payload_contains_resume_contract() -> None:
    payload, _model, _optimizer, _scheduler = make_payload()

    assert {
        "format_version",
        "kind",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "epoch",
        "step_in_epoch",
        "global_step",
        "config",
        "seed",
        "world_size",
        "rng_states",
        "loss",
        "learning_rate",
        "aggregate_k_gradients",
        "source",
        "started_at",
        "created_at",
    } <= payload.keys()
    assert payload["format_version"] == 1
    assert payload["kind"] == "focat_training"
    assert payload["global_step"] == 3


def test_atomic_checkpoint_updates_latest_and_retains_newest(tmp_path: Path) -> None:
    newest = None
    for step in range(1, 5):
        payload, _model, _optimizer, _scheduler = make_payload(step)
        newest = save_training_checkpoint(payload, tmp_path, keep_last=2)

    assert newest == tmp_path / "step_00000004.pt"
    assert sorted(path.name for path in tmp_path.glob("step_*.pt")) == [
        "step_00000003.pt",
        "step_00000004.pt",
    ]
    latest = tmp_path / "latest.pt"
    assert latest.is_file()
    assert latest.samefile(newest)
    loaded = torch.load(latest, map_location="cpu", weights_only=False)
    assert loaded["global_step"] == 4


def test_training_checkpoint_restores_all_training_state(tmp_path: Path) -> None:
    payload, saved_model, saved_optimizer, saved_scheduler = make_payload()
    checkpoint = save_training_checkpoint(payload, tmp_path, keep_last=2)
    expected_scheduler = saved_scheduler.state_dict()

    restored_model, restored_optimizer, restored_scheduler = make_optimizer_state(1)
    restored = load_training_checkpoint(
        checkpoint,
        model=restored_model,
        optimizer=restored_optimizer,
        scheduler=restored_scheduler,
        expected_world_size=1,
        rank=0,
        restore_rng=False,
    )

    assert restored.epoch == 2
    assert restored.step_in_epoch == 7
    assert restored.global_step == 3
    assert restored.aggregate_k_gradients == 1
    assert optimizer_steps(restored_optimizer) == {3}
    assert restored_scheduler.state_dict() == expected_scheduler
    for expected, actual in zip(saved_model.parameters(), restored_model.parameters()):
        torch.testing.assert_close(actual, expected)


def test_training_checkpoint_rejects_world_size_change(tmp_path: Path) -> None:
    payload, _saved_model, _saved_optimizer, _saved_scheduler = make_payload()
    checkpoint = save_training_checkpoint(payload, tmp_path, keep_last=2)
    model, optimizer, scheduler = make_optimizer_state(1)

    with pytest.raises(ValueError, match="world size"):
        load_training_checkpoint(
            checkpoint,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            expected_world_size=2,
            rank=0,
            restore_rng=False,
        )


def tiny_focat_config():
    config = get_model_default_config("mothernet")
    config["device"] = "cpu"
    config["orchestration"] = {
        "experiment": "checkpoint tests",
        "progress_bar": False,
    }
    config["prior"]["num_features"] = 2
    config["transformer"].update(
        {
            "emsize": 16,
            "nlayers": 1,
            "nhid_factor": 2,
            "nhead": 2,
            "y_encoder": "linear",
        }
    )
    config["mothernet"].update(
        {
            "weight_embedding_rank": 2,
            "predicted_hidden_layer_size": 8,
            "decoder_embed_dim": 16,
            "decoder_hidden_size": 32,
            "predicted_hidden_layers": 1,
            "decoder_hidden_layers": 1,
            "bins_output_n": 16,
            "bins_to_sum_n": 2,
        }
    )
    return config


@pytest.mark.parametrize("new_format", [False, True])
def test_inference_loader_accepts_legacy_and_training_checkpoints(
    tmp_path: Path, new_format: bool
) -> None:
    config = tiny_focat_config()
    _loss, model, _loader, _epoch = get_model(
        config, device="cpu", should_train=False, verbose=False
    )
    if new_format:
        stored = {
            "format_version": 1,
            "kind": "focat_training",
            "model_state": model.state_dict(),
            "config": config,
        }
    else:
        stored = (model.state_dict(), None, None, config)
    checkpoint = tmp_path / ("new.pt" if new_format else "legacy.pt")
    torch.save(stored, checkpoint)

    restored, restored_config = load_model(checkpoint, device="cpu")

    assert restored_config["model_type"] == "mothernet"
    for expected, actual in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(actual, expected)
