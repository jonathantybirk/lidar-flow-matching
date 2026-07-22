import argparse
import copy
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from lidar_flow.config import load_experiment_config
from lidar_flow.dataset import CompletionDataset
from lidar_flow.flow import completion_loss
from lidar_flow.model import PointFlowTransformer

ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LiDAR point completion")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "train.toml")
    parser.add_argument(
        "--data", type=Path, default=ROOT / "data" / "processed" / "03001627"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "runs" / "chairs")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--run")
    return parser.parse_args()


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def move(
    batch: dict[str, torch.Tensor], target: torch.device
) -> dict[str, torch.Tensor]:
    return {name: value.to(target, non_blocking=True) for name, value in batch.items()}


def update_ema(
    ema: PointFlowTransformer, model: PointFlowTransformer, decay: float
) -> None:
    with torch.no_grad():
        for average, current in zip(ema.parameters(), model.parameters(), strict=True):
            average.lerp_(current, 1 - decay)
        for average, current in zip(ema.buffers(), model.buffers(), strict=True):
            average.copy_(current)


def learning_rate_multiplier(
    step: int, warmup: int, total: int, minimum: float
) -> float:
    if step < warmup:
        return (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


@torch.no_grad()
def validate(
    model: PointFlowTransformer,
    loader: DataLoader,
    target_device: torch.device,
    batches: int,
    chamfer_weight: float,
    observation_weight: float,
    seed: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {
        name: 0.0 for name in ("loss", "flow", "chamfer", "observation")
    }
    generator = torch.Generator(device=target_device).manual_seed(seed)
    for index, batch in enumerate(loader):
        if index == batches:
            break
        batch = move(batch, target_device)
        origin = torch.randn(
            batch["target"].shape, generator=generator, device=target_device
        )
        time = torch.rand(len(origin), generator=generator, device=target_device)
        loss, terms = completion_loss(
            model, batch, chamfer_weight, observation_weight, origin, time
        )
        totals["loss"] += loss.item()
        for name, value in terms.items():
            totals[name] += value.item()
    return {name: value / batches for name, value in totals.items()}


def save_checkpoint(
    path: Path,
    model: PointFlowTransformer,
    ema: PointFlowTransformer,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    step: int,
    config: dict,
) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "ema": ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "config": config,
        },
        temporary,
    )
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    config = load_experiment_config(args.config)
    training = config.training
    batch_size = args.batch_size or training.batch_size
    steps = args.steps or training.steps
    workers = training.workers if args.workers is None else args.workers
    random.seed(training.seed)
    np.random.seed(training.seed)
    torch.manual_seed(training.seed)
    target_device = device()
    args.output.mkdir(parents=True, exist_ok=True)

    datasets = {
        split: CompletionDataset(
            args.data,
            split,
            config.model.condition_points,
            config.model.points,
        )
        for split in ("train", "validation")
    }
    loaders = {
        split: DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=workers,
            pin_memory=target_device.type == "cuda",
            persistent_workers=workers > 0,
            drop_last=split == "train",
        )
        for split, dataset in datasets.items()
    }
    model = PointFlowTransformer(config.model).to(target_device)
    ema = copy.deepcopy(model).eval()
    for parameter in ema.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=training.learning_rate,
        betas=(0.9, 0.95),
        weight_decay=training.weight_decay,
    )
    minimum = training.minimum_learning_rate / training.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(
            step, training.warmup_steps, steps, minimum
        ),
    )
    start = 0
    if args.resume:
        checkpoint = torch.load(
            args.resume, map_location=target_device, weights_only=True
        )
        model.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start = checkpoint["step"]

    training_values = asdict(config.training)
    training_values.update(batch_size=batch_size, steps=steps, workers=workers)
    configuration = {"model": asdict(config.model), "training": training_values}
    (args.output / "config.json").write_text(
        json.dumps(configuration, indent=2) + "\n", encoding="utf-8"
    )
    tracking = (
        wandb.init(
            project="lidar-flow-matching",
            name=args.run,
            config=configuration,
        )
        if args.run
        else None
    )
    iterator = iter(loaders["train"])
    model.train()
    running = {name: 0.0 for name in ("loss", "flow", "chamfer", "observation")}
    last_log = time.perf_counter()
    progress = tqdm(
        range(start + 1, steps + 1),
        total=steps,
        initial=start,
        desc=f"Training on {target_device.type}",
        unit="step",
        dynamic_ncols=True,
        disable=not sys.stderr.isatty(),
    )
    for step in progress:
        tracking_metrics = {}
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loaders["train"])
            batch = next(iterator)
        batch = move(batch, target_device)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=target_device.type,
            dtype=torch.bfloat16,
            enabled=target_device.type == "cuda",
        ):
            loss, terms = completion_loss(
                model,
                batch,
                training.chamfer_weight,
                training.observation_weight,
            )
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), training.gradient_clip
        )
        optimizer.step()
        scheduler.step()
        update_ema(ema, model, training.ema_decay)
        loss_value = loss.item()
        running["loss"] += loss_value
        for name, value in terms.items():
            running[name] += value.item()
        progress.set_postfix(
            loss=f"{loss_value:.4f}",
            lr=f"{scheduler.get_last_lr()[0]:.2e}",
            refresh=False,
        )

        if step % training.log_every == 0:
            now = time.perf_counter()
            tracking_metrics = {
                f"train/{name}": value / training.log_every
                for name, value in running.items()
            }
            tracking_metrics.update(
                {
                    "train/gradient_norm": gradient_norm.item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/steps_per_second": training.log_every / (now - last_log),
                }
            )
            if target_device.type == "cuda":
                tracking_metrics["train/peak_memory_gb"] = (
                    torch.cuda.max_memory_allocated() / 2**30
                )
            progress.write(
                json.dumps(
                    {
                        "step": step,
                        "loss": tracking_metrics["train/loss"],
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                )
            )
            running = {name: 0.0 for name in running}
            last_log = now
        if step % training.validate_every == 0 or step == steps:
            model_metrics = validate(
                model,
                loaders["validation"],
                target_device,
                training.validation_batches,
                training.chamfer_weight,
                training.observation_weight,
                training.seed,
            )
            ema_metrics = validate(
                ema,
                loaders["validation"],
                target_device,
                training.validation_batches,
                training.chamfer_weight,
                training.observation_weight,
                training.seed,
            )
            progress.write(
                json.dumps(
                    {
                        "step": step,
                        "validation": {"model": model_metrics, "ema": ema_metrics},
                    }
                )
            )
            tracking_metrics.update(
                {
                    **{
                        f"validation/model/{name}": value
                        for name, value in model_metrics.items()
                    },
                    **{
                        f"validation/ema/{name}": value
                        for name, value in ema_metrics.items()
                    },
                }
            )
            model.train()
            last_log = time.perf_counter()
        if step % training.checkpoint_every == 0 or step == steps:
            save_checkpoint(
                args.output / "checkpoint.pt",
                model,
                ema,
                optimizer,
                scheduler,
                step,
                configuration,
            )
        if tracking and tracking_metrics:
            tracking.log(tracking_metrics, step=step)
    if tracking:
        tracking.finish()


if __name__ == "__main__":
    main()
