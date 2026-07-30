import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import default_collate

from lidar_flow.config import ModelConfig
from lidar_flow.dataset import ViewDataset
from lidar_flow.flow import sample
from lidar_flow.geometry import points_from_state
from lidar_flow.model import LidarFlowTransformer, device
from lidar_flow.visualize import draw_mesh, draw_sensor, object_mesh, set_equal_axes
from train import move

ROOT = Path(__file__).parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review generated views")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/vehicles")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--splits", nargs="+", default=("validation", "test"))
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--synset", nargs="+", default=(), help="cars dominate an even draw"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/view-review")
    return parser.parse_args()


def review(
    model: LidarFlowTransformer,
    dataset: ViewDataset,
    index: int,
    split: str,
    args: argparse.Namespace,
    target_device: torch.device,
) -> dict[str, float]:
    path, source_view, views = dataset.items[index]
    item = dataset[index]
    target_view = (source_view + 1 + views // 3) % views
    generated = sample(
        model, move(default_collate([item]), target_device), steps=args.steps
    )[0].cpu().numpy()

    with np.load(path) as stored:
        poses = stored["object_from_sensors"]
        mesh = object_mesh(args.raw, path.stem, stored["object_from_original"])

    def scan(state: np.ndarray, view: int) -> np.ndarray:
        points = points_from_state(
            state, dataset.ray_directions, dataset.range_center, dataset.range_scale
        )
        return points @ poses[view][:3, :3].T + poses[view][:3, 3]

    current = scan(item["current"], source_view)
    target = scan(item["target"], target_view)
    predicted = scan(generated, target_view)

    predicted_hits = generated[0] > 0
    target_hits = item["target"][0] > 0
    both = predicted_hits & target_hits
    iou = both.sum() / (predicted_hits | target_hits).sum()
    rmse = np.sqrt(np.mean((generated[1][both] - item["target"][1][both]) ** 2))

    figure = plt.figure(figsize=(11, 10), constrained_layout=True)
    axes = [
        figure.add_subplot(2, 2, position, projection="3d") for position in range(1, 5)
    ]
    draw_mesh(axes[0], mesh)
    for axis, points, view, color, title in (
        (axes[1], target, target_view, "tab:blue", "Target"),
        (axes[2], current, source_view, "tab:orange", "Input"),
        (axes[3], predicted, target_view, "tab:green", "Predicted"),
    ):
        axis.scatter(*points.T, s=11, color=color, depthshade=False)
        draw_sensor(axis, poses[view])
        axis.set_title(f"{title} scan, view {view} ({len(points)} returns)")
    axes[3].set_title(
        f"Predicted scan, view {target_view} ({len(predicted)} returns)\n"
        f"IoU {iou:.3f}   range RMSE {rmse:.3f}"
    )

    bounds = np.vstack(
        (
            mesh.vertices,
            current,
            target,
            predicted,
            poses[source_view][:3, 3],
            poses[target_view][:3, 3],
        )
    )
    for axis in axes:
        set_equal_axes(axis, bounds)

    figure.suptitle(f"{split}  ·  {path.stem}", fontsize=11)
    args.output.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        args.output / f"{split}_{index:06d}_{path.stem}.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close(figure)
    return {"split": split, "index": index, "iou": float(iou), "rmse": float(rmse)}


def main() -> None:
    args = parse_args()
    target_device = device()
    torch.manual_seed(args.seed)
    checkpoint = torch.load(
        args.checkpoint, map_location=target_device, weights_only=True
    )
    generator = np.random.default_rng(args.seed)

    results = []
    for split in args.splits:
        dataset = ViewDataset(args.data, split)
        model = LidarFlowTransformer(
            ModelConfig(**checkpoint["config"]["model"]), dataset.ray_directions
        ).to(target_device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        candidates = [
            index
            for index, (path, _, _) in enumerate(dataset.items)
            if not args.synset or path.stem.split("_", 1)[0] in args.synset
        ]
        for index in sorted(generator.choice(candidates, args.samples, replace=False)):
            results.append(
                review(model, dataset, int(index), split, args, target_device)
            )
            print(json.dumps(results[-1]), flush=True)

    print(f"\nwrote {len(results)} figures to {args.output}")
    for split in args.splits:
        rows = [result for result in results if result["split"] == split]
        print(
            f"{split:>10}: mean IoU {np.mean([row['iou'] for row in rows]):.3f}  "
            f"mean range RMSE {np.mean([row['rmse'] for row in rows]):.3f}"
        )


if __name__ == "__main__":
    main()
