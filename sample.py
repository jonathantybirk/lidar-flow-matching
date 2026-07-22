import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import default_collate

from lidar_flow.config import ModelConfig
from lidar_flow.dataset import CompletionDataset
from lidar_flow.flow import sample
from lidar_flow.model import PointFlowTransformer
from lidar_flow.visualize import set_equal_axes
from train import device, move

ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample validation completions")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/03001627")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--weights", choices=("model", "ema"), default="ema")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/completions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_device = device()
    checkpoint = torch.load(
        args.checkpoint, map_location=target_device, weights_only=True
    )
    config = ModelConfig(**checkpoint["config"]["model"])
    model = PointFlowTransformer(config).to(target_device).eval()
    model.load_state_dict(checkpoint[args.weights])
    dataset = CompletionDataset(
        args.data, "validation", config.condition_points, config.points
    )
    batch = default_collate([dataset[args.index]])
    repeated = {
        name: value.repeat_interleave(args.samples, dim=0)
        for name, value in move(batch, target_device).items()
        if name != "target"
    }
    generated = sample(model, repeated, steps=args.steps).cpu().numpy()
    observed = batch["observed_xyz"][0, batch["observed_mask"][0]].numpy()
    sensor = batch["sensor_features"][0, :3].numpy()
    forward = batch["sensor_features"][0, 3:6].numpy()
    target = batch["target"][0].numpy()

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.checkpoint.stem}_{args.weights}_{args.index:05d}"
    np.savez(
        args.output / f"{stem}.npz",
        generated=generated,
        observed=observed,
        target=target,
    )
    figure = plt.figure(figsize=(5 * (args.samples + 1), 5))
    arrow_end = sensor + 0.3 * forward
    bounds = np.concatenate(
        (target, observed, generated.reshape(-1, 3), sensor[None], arrow_end[None])
    )
    axis = figure.add_subplot(1, args.samples + 1, 1, projection="3d")
    axis.scatter(*observed.T, s=8, color="tab:orange")
    axis.scatter(*sensor, s=75, color="tab:red")
    axis.quiver(
        *sensor,
        *forward,
        length=0.3,
        color="tab:red",
        arrow_length_ratio=0.22,
    )
    axis.set_title("LiDAR observation")
    set_equal_axes(axis, bounds)
    for index, completion in enumerate(generated, start=1):
        axis = figure.add_subplot(1, args.samples + 1, index + 1, projection="3d")
        axis.scatter(*completion.T, s=5, color="tab:blue")
        axis.set_title(f"Draw {index}")
        set_equal_axes(axis, bounds)
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(args.output / f"{stem}.png", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
