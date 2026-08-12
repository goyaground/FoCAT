from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import random
import socket
import subprocess
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from ticl.checkpointing import (
    capture_rng_state,
    checkpoint_payload,
    load_training_checkpoint,
    save_training_checkpoint,
)
from ticl.dataloader import get_dataloader
from ticl.model_builder import get_criterion, get_model
from ticl.model_configs import get_model_default_config
from ticl.train import eval_criterion


UPSTREAM_URL = "https://github.com/NTAILab/FoCAT.git"
UPSTREAM_BASE_COMMIT = "608006595eaf2eda40c36f097c369d1efd65d500"


@dataclass(frozen=True)
class TrainingOptions:
    profile: str = "full"
    checkpoint_dir: Path = Path("checkpoints/focat")
    resume: Path | None = None
    seed: int = 42
    max_global_steps: int | None = None
    checkpoint_every_steps: int = 1000
    keep_last: int = 3
    log_every_steps: int = 10
    device: str = "auto"
    cpu_threads: int = 8


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    initialized_here: bool


def resolve_config(profile: str, world_size: int) -> dict[str, Any]:
    if profile not in {"full", "smoke"}:
        raise ValueError(f"unknown training profile: {profile}")
    if world_size < 1:
        raise ValueError("world_size must be positive")
    config = get_model_default_config("mothernet")
    if not config["transformer"]["classification_task"]:
        config["prior"]["classification"]["max_num_classes"] = 0
        config["transformer"]["y_encoder"] = "linear"
    repository_global_batch = int(config["dataloader"]["batch_size"])
    if profile == "full":
        if repository_global_batch % world_size != 0:
            raise ValueError(
                f"world size {world_size} cannot preserve global batch size "
                f"{repository_global_batch}"
            )
        per_rank_batch = repository_global_batch // world_size
        global_batch = repository_global_batch
    else:
        per_rank_batch = 1
        global_batch = world_size
        config["prior"].update(
            {
                "num_features": 4,
                "n_samples": 40,
                "eval_positions": [32],
            }
        )
        config["dataloader"].update(
            {
                "num_steps": 2,
                "min_eval_pos": 32,
                "random_n_samples": 0,
                "n_test_samples": 0,
            }
        )
        config["transformer"].update(
            {
                "emsize": 32,
                "nlayers": 1,
                "nhid_factor": 2,
                "nhead": 4,
                "y_encoder": "linear",
                "dropout": 0.0,
            }
        )
        config["mothernet"].update(
            {
                "weight_embedding_rank": 4,
                "predicted_hidden_layer_size": 16,
                "decoder_embed_dim": 32,
                "decoder_hidden_size": 64,
                "predicted_hidden_layers": 1,
                "decoder_hidden_layers": 1,
            }
        )
        config["optimizer"].update(
            {
                "epochs": 2,
                "stop_after_epochs": None,
                "aggregate_k_gradients": 1,
                "adaptive_batch_size": False,
                "train_mixed_precision": False,
            }
        )
    config["dataloader"]["batch_size"] = per_rank_batch
    config["device"] = "cpu"
    config["num_gpus"] = world_size
    config["orchestration"] = {
        "experiment": f"FoCAT {profile} pretraining",
        "progress_bar": False,
    }
    config["runtime"] = {
        "profile": profile,
        "world_size": world_size,
        "repository_global_batch_size": repository_global_batch,
        "global_batch_size": global_batch,
        "batch_size_per_rank": per_rank_batch,
    }
    return config


def _setup_distributed(device_name: str) -> DistributedContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    initialized_here = False
    if world_size > 1:
        if device_name == "cpu" or not torch.cuda.is_available():
            raise RuntimeError("multi-process FoCAT training requires CUDA/NCCL")
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(
                backend="nccl",
                init_method="env://",
                timeout=timedelta(minutes=30),
            )
            initialized_here = True
        device = torch.device("cuda", local_rank)
    elif device_name == "cpu":
        device = torch.device("cpu")
    elif device_name in {"auto", "cuda"}:
        if not torch.cuda.is_available():
            if device_name == "cuda":
                raise RuntimeError("CUDA was requested but is unavailable")
            device = torch.device("cpu")
        else:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
    else:
        device = torch.device(device_name)
    return DistributedContext(rank, local_rank, world_size, device, initialized_here)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _build_scheduler(optimizer, config):
    optimizer_config = config["optimizer"]
    schedule = optimizer_config["learning_rate_schedule"]
    if schedule != "cosine":
        raise ValueError(f"pretrain.py currently supports repository cosine schedule, got {schedule}")
    base_scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=100,
        eta_min=optimizer_config["min_lr"],
    )
    warmup_epochs = int(optimizer_config["warmup_epochs"])
    if warmup_epochs == 0:
        return base_scheduler
    return SequentialLR(
        optimizer,
        [
            LinearLR(
                optimizer,
                start_factor=1e-2,
                end_factor=1.0,
                total_iters=warmup_epochs,
            ),
            base_scheduler,
        ],
        milestones=[warmup_epochs],
    )


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)


def _log(writer_rank: int, event: str, **values) -> None:
    if writer_rank != 0:
        return
    record = {"event": event, **values}
    print(json.dumps(_json_safe(record), sort_keys=True), flush=True)


def _runtime_source() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    runtime_commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    return {
        "upstream_url": UPSTREAM_URL,
        "upstream_base_commit": UPSTREAM_BASE_COMMIT,
        "runtime_commit": runtime_commit,
        "runtime_dirty": dirty,
    }


def _adaptive_accumulation(
    aggregate: int,
    stage: int,
    completed_epoch: int,
    loss: float,
) -> tuple[int, int]:
    if stage == 0 and completed_epoch >= 20:
        return aggregate * 2, 1
    if stage == 1 and completed_epoch >= 50:
        return aggregate * 2, 2
    if stage == 2 and completed_epoch >= 200:
        return aggregate * 2, 3
    if stage == 3 and loss >= 1000:
        return aggregate * 2, 4
    return aggregate, stage


def _gather_rng_states(context: DistributedContext) -> list[dict[str, Any]]:
    local_state = capture_rng_state()
    if context.world_size == 1:
        return [local_state]
    gathered = [None for _ in range(context.world_size)]
    dist.all_gather_object(gathered, local_state)
    return gathered


def _should_checkpoint(
    global_step: int,
    checkpoint_every_steps: int,
    stopping: bool,
) -> bool:
    return (
        global_step == 1
        or global_step % checkpoint_every_steps == 0
        or stopping
    )


def _load_resume_payload(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or payload.get("kind") != "focat_training":
        raise ValueError("--resume requires a FoCAT dictionary training checkpoint")
    return payload


def run_training(options: TrainingOptions) -> dict[str, Any]:
    if options.checkpoint_every_steps < 1 or options.log_every_steps < 1:
        raise ValueError("checkpoint and log frequencies must be positive")
    if options.max_global_steps is not None and options.max_global_steps < 1:
        raise ValueError("max_global_steps must be positive")
    context = _setup_distributed(options.device)
    torch.set_num_threads(options.cpu_threads)
    rank_seed = options.seed + context.rank
    _seed_everything(rank_seed)
    started_at = datetime.now(timezone.utc).isoformat()
    source = _runtime_source()
    try:
        config = resolve_config(options.profile, context.world_size)
        resume_payload = None
        if options.resume is not None:
            resume_payload = _load_resume_payload(Path(options.resume))
            if int(resume_payload["seed"]) != options.seed:
                raise ValueError("resume seed must match checkpoint seed")
            if int(resume_payload["world_size"]) != context.world_size:
                raise ValueError("resume world size must match checkpoint world size")
            saved_profile = resume_payload["config"].get("runtime", {}).get("profile")
            if saved_profile != options.profile:
                raise ValueError("resume profile must match checkpoint profile")
            config = copy.deepcopy(resume_payload["config"])
            started_at = str(resume_payload.get("started_at", started_at))
            source = copy.deepcopy(resume_payload.get("source", source))
        config["device"] = str(context.device)
        config["num_gpus"] = context.world_size
        _loss, model, _loader, _epoch = get_model(
            config,
            device=str(context.device),
            should_train=False,
            verbose=context.rank == 0,
        )
        model.to(context.device)
        if context.world_size > 1:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank],
                output_device=context.local_rank,
                broadcast_buffers=False,
            )
        unwrapped = model.module if hasattr(model, "module") else model
        optimizer_config = config["optimizer"]
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=optimizer_config["learning_rate"],
            weight_decay=optimizer_config["weight_decay"],
            betas=(optimizer_config["adam_beta1"], 0.999),
        )
        scheduler = _build_scheduler(optimizer, config)
        criterion = get_criterion(
            config["prior"]["classification"]["max_num_classes"],
            device=context.device,
            total_num_bins=config["mothernet"]["bins_output_n"],
            num_bins_to_sum=config["mothernet"]["bins_to_sum_n"],
        )
        loader = get_dataloader(
            config["prior"],
            config["dataloader"],
            device="cpu",
        )
        epoch = 1
        step_in_epoch = 0
        global_step = 0
        aggregate = int(optimizer_config["aggregate_k_gradients"])
        adaptive_stage = 0
        latest_loss = float("inf")
        if resume_payload is not None:
            resumed = load_training_checkpoint(
                options.resume,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                expected_world_size=context.world_size,
                rank=context.rank,
                restore_rng=True,
            )
            epoch = resumed.epoch
            step_in_epoch = resumed.step_in_epoch
            global_step = resumed.global_step
            aggregate = resumed.aggregate_k_gradients
            latest_loss = resumed.loss
            adaptive_stage = int(resume_payload.get("adaptive_batch_stage", 0))
            if step_in_epoch >= int(config["dataloader"]["num_steps"]):
                epoch += 1
                step_in_epoch = 0
        _log(
            context.rank,
            "run_start",
            started_at=started_at,
            hostname=socket.gethostname(),
            pid=os.getpid(),
            rank=context.rank,
            local_rank=context.local_rank,
            world_size=context.world_size,
            device=str(context.device),
            physical_cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
            seed=options.seed,
            rank_seed=rank_seed,
            python_version=os.sys.version.split()[0],
            torch_version=torch.__version__,
            cuda_version=torch.version.cuda,
            source=source,
            config=config,
            resume=str(options.resume) if options.resume else None,
            resumed_global_step=global_step,
        )
        epochs = int(optimizer_config["epochs"])
        num_steps = int(config["dataloader"]["num_steps"])
        stop_requested = (
            options.max_global_steps is not None
            and global_step >= options.max_global_steps
        )
        while epoch <= epochs and not stop_requested:
            if num_steps % aggregate != 0:
                raise ValueError(
                    f"num_steps {num_steps} must be divisible by gradient "
                    f"accumulation {aggregate}"
                )
            loader.epoch_count = epoch - 1
            iterator = iter(loader)
            accumulated_loss = 0.0
            microbatches = 0
            for batch_index in range(step_in_epoch + 1, num_steps + 1):
                data, targets, single_eval_pos = next(iterator)
                microbatches += 1
                accumulation_boundary = batch_index % aggregate == 0
                sync_context = (
                    nullcontext()
                    if context.world_size == 1 or accumulation_boundary
                    else model.no_sync()
                )
                with sync_context:
                    x, y, treatment = data
                    output = model(
                        (
                            x.to(context.device),
                            y.to(context.device),
                            treatment.to(context.device),
                        ),
                        single_eval_pos=single_eval_pos,
                    )
                    loss, _nan_share = eval_criterion(
                        criterion,
                        targets[single_eval_pos:],
                        output,
                        device=context.device,
                        n_out=unwrapped.n_out,
                    )
                    if not torch.isfinite(loss):
                        raise FloatingPointError(
                            f"non-finite loss at epoch={epoch} batch={batch_index}: {loss}"
                        )
                    accumulated_loss += float(loss.detach().cpu())
                    (loss / aggregate).backward()
                step_in_epoch = batch_index
                if not accumulation_boundary:
                    continue
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    1.0,
                    foreach=True,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                reduced_loss = torch.tensor(
                    accumulated_loss / microbatches,
                    device=context.device,
                    dtype=torch.float64,
                )
                if context.world_size > 1:
                    dist.all_reduce(reduced_loss, op=dist.ReduceOp.SUM)
                    reduced_loss /= context.world_size
                latest_loss = float(reduced_loss.cpu())
                accumulated_loss = 0.0
                microbatches = 0
                epoch_finished = step_in_epoch == num_steps
                if epoch_finished:
                    scheduler.step()
                    if optimizer_config["adaptive_batch_size"]:
                        aggregate, adaptive_stage = _adaptive_accumulation(
                            aggregate,
                            adaptive_stage,
                            epoch,
                            latest_loss,
                        )
                current_lr = float(optimizer.param_groups[0]["lr"])
                stopping = (
                    options.max_global_steps is not None
                    and global_step >= options.max_global_steps
                )
                if global_step % options.log_every_steps == 0 or stopping:
                    _log(
                        context.rank,
                        "train_step",
                        epoch=epoch,
                        step_in_epoch=step_in_epoch,
                        global_step=global_step,
                        loss=latest_loss,
                        learning_rate=current_lr,
                        gradient_norm=float(gradient_norm.detach().cpu()),
                        aggregate_k_gradients=aggregate,
                    )
                if _should_checkpoint(
                    global_step,
                    options.checkpoint_every_steps,
                    stopping,
                ):
                    rng_states = _gather_rng_states(context)
                    if context.rank == 0:
                        payload = checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            scheduler=scheduler,
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                            global_step=global_step,
                            config=config,
                            seed=options.seed,
                            world_size=context.world_size,
                            rng_states=rng_states,
                            loss=latest_loss,
                            learning_rate=current_lr,
                            aggregate_k_gradients=aggregate,
                            source=source,
                            started_at=started_at,
                        )
                        payload["adaptive_batch_stage"] = adaptive_stage
                        path = save_training_checkpoint(
                            payload,
                            options.checkpoint_dir,
                            keep_last=options.keep_last,
                        )
                        _log(
                            context.rank,
                            "checkpoint_saved",
                            path=str(path.resolve()),
                            epoch=epoch,
                            step_in_epoch=step_in_epoch,
                            global_step=global_step,
                        )
                    if context.world_size > 1:
                        dist.barrier()
                if stopping:
                    stop_requested = True
                    break
            if step_in_epoch >= num_steps:
                epoch += 1
                step_in_epoch = 0
        summary_epoch = epoch - 1 if step_in_epoch == 0 and epoch > 1 else epoch
        summary_step = num_steps if step_in_epoch == 0 and epoch > 1 else step_in_epoch
        summary = {
            "status": "stopped_at_limit" if stop_requested else "training_complete",
            "epoch": summary_epoch,
            "step_in_epoch": summary_step,
            "global_step": global_step,
            "loss": latest_loss,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "world_size": context.world_size,
            "device": str(context.device),
            "started_at": started_at,
        }
        _log(context.rank, "run_stop", **summary)
        return summary
    finally:
        if context.initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def parse_args(argv: list[str] | None = None) -> TrainingOptions:
    parser = argparse.ArgumentParser(description="FoCAT full pretraining")
    parser.add_argument("--profile", choices=("full", "smoke"), default="full")
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints/focat"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-global-steps", type=int)
    parser.add_argument("--checkpoint-every-steps", type=int, default=1000)
    parser.add_argument("--keep-last", type=int, default=3)
    parser.add_argument("--log-every-steps", type=int, default=10)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cpu-threads", type=int, default=8)
    return TrainingOptions(**vars(parser.parse_args(argv)))


def main(argv: list[str] | None = None) -> int:
    options = parse_args(argv)
    summary = run_training(options)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(_json_safe(asdict(options) | summary), sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
