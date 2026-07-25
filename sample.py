import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import default_collate

from lidar_flow.config import ModelConfig
from lidar_flow.dataset import ViewDataset
from lidar_flow.flow import sample
from lidar_flow.geometry import points_from_state
from lidar_flow.model import LidarFlowTransformer
from lidar_flow.visualize import set_equal_axes
from train import device, move

ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample future LiDAR views")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/vehicles")
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/view-flow")
    return parser.parse_args()


def transform(points: np.ndarray, action: np.ndarray) -> np.ndarray:
    rotation = action[:9].reshape(3, 3)
    return points @ rotation.T + action[9:]


def draw_sensor(axis, position: np.ndarray, forward: np.ndarray) -> None:
    axis.scatter(*position, s=70, color="tab:red")
    axis.quiver(
        *position,
        *forward,
        length=0.35,
        color="tab:red",
        arrow_length_ratio=0.22,
    )


def main() -> None:
    args = parse_args()
    target_device = device()
    checkpoint = torch.load(
        args.checkpoint, map_location=target_device, weights_only=True
    )
    config = ModelConfig(**checkpoint["config"]["model"])
    dataset = ViewDataset(args.data, args.split)
    model = LidarFlowTransformer(config, dataset.ray_directions).to(target_device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    item = dataset[args.index]
    batch = move(default_collate([item]), target_device)
    repeated = {
        name: value.repeat_interleave(args.samples, dim=0)
        for name, value in batch.items()
    }
    generated = sample(model, repeated, steps=args.steps).cpu().numpy()
    action = item["action"]
    source = points_from_state(
        item["current"],
        dataset.ray_directions,
        dataset.range_center,
        dataset.range_scale,
    )
    target = transform(
        points_from_state(
            item["target"],
            dataset.ray_directions,
            dataset.range_center,
            dataset.range_scale,
        ),
        action,
    )
    predictions = [
        transform(
            points_from_state(
                state,
                dataset.ray_directions,
                dataset.range_center,
                dataset.range_scale,
            ),
            action,
        )
        for state in generated
    ]
    source_position = np.zeros(3, dtype=np.float32)
    source_forward = np.array((0.0, 0.0, 1.0), dtype=np.float32)
    target_position = action[9:]
    target_forward = action[:9].reshape(3, 3)[:, 2]

    args.output.mkdir(parents=True, exist_ok=True)
    stem = f"{args.split}_{args.index:05d}_{args.checkpoint.stem}"
    np.savez(
        args.output / f"{stem}.npz",
        current=item["current"],
        target=item["target"],
        generated=generated,
        action=action,
    )
    for draw, prediction in enumerate(predictions, start=1):
        bounds = np.concatenate(
            (
                source,
                target,
                prediction,
                source_position[None],
                target_position[None],
            )
        )
        figure = plt.figure(figsize=(15, 5), constrained_layout=True)
        axes = [figure.add_subplot(1, 3, index, projection="3d") for index in range(1, 4)]
        axes[0].scatter(*source.T, s=10, color="tab:orange")
        draw_sensor(axes[0], source_position, source_forward)
        axes[0].set_title(f"Current observation ({len(source)} returns)")
        axes[1].scatter(*target.T, s=10, color="tab:blue")
        draw_sensor(axes[1], target_position, target_forward)
        axes[1].set_title(f"Target observation ({len(target)} returns)")
        axes[2].scatter(*prediction.T, s=10, color="tab:green")
        draw_sensor(axes[2], target_position, target_forward)
        axes[2].set_title(f"Generated observation ({len(prediction)} returns)")
        for axis in axes:
            set_equal_axes(axis, bounds)
        figure.savefig(
            args.output / f"{stem}_{draw:02d}.png",
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.15,
        )
        plt.close(figure)


if __name__ == "__main__":
    main()
