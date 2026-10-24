import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from lidar_flow.dataset import ViewDataset, evaluation_target
from lidar_flow.flow import sample
from lidar_flow.model import device, load_model

ROOT = Path(__file__).parent
COUNTS = (
    "intersection",
    "union",
    "predicted_hits",
    "target_hits",
    "squared_range_error_on_target_hits",
    "squared_range_error_on_shared_hits",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate on every test view pair")
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=ROOT / "runs/view-flow-a100-qknorm/best.pt",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/vehicles")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, help="evaluate only the first pairs")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/evaluation")
    return parser.parse_args()


def counts(predicted: torch.Tensor, target: torch.Tensor) -> np.ndarray:
    """Per-pair occupancy and range-error sums, one column per name in COUNTS."""
    predicted, target = predicted.cpu().double(), target.cpu().double()
    predicted_hits, target_hits = predicted[:, 0] > 0, target[:, 0] > 0
    shared = predicted_hits & target_hits
    squared_error = (predicted[:, 1] - target[:, 1]).square()
    return torch.stack(
        [
            shared.sum((1, 2)),
            (predicted_hits | target_hits).sum((1, 2)),
            predicted_hits.sum((1, 2)),
            target_hits.sum((1, 2)),
            (squared_error * target_hits).sum((1, 2)),
            (squared_error * shared).sum((1, 2)),
        ],
        dim=1,
    ).numpy()


def metrics(rows: list[list[float]]) -> dict[str, float]:
    """Micro-average over all rays, matching validation in train.py."""
    shared, union, predicted, target, error, shared_error = np.sum(rows, axis=0)
    return {
        "occupancy_iou": shared / union,
        "occupancy_precision": shared / predicted,
        "occupancy_recall": shared / target,
        "range_rmse": math.sqrt(error / target),
        "range_rmse_on_shared_hits": math.sqrt(shared_error / shared),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    target_device = device()
    dataset = ViewDataset(args.data, args.split)
    model, checkpoint = load_model(args.checkpoint, dataset.ray_directions, target_device)
    pairs = range(min(args.limit or len(dataset), len(dataset)))
    loader = DataLoader(Subset(dataset, pairs), batch_size=args.batch_size)
    # Noise is drawn on the CPU so every device sees the same samples.
    generator = torch.Generator().manual_seed(args.seed)

    args.output.mkdir(parents=True, exist_ok=True)
    totals = {"model": [], "copy_input": []}
    with (args.output / "per_view.jsonl").open("w") as stream:
        index = 0
        for batch in loader:
            batch = {name: value.to(target_device) for name, value in batch.items()}
            origin = torch.randn(batch["target"].shape, generator=generator)
            predicted = sample(model, batch, args.steps, origin.to(target_device))
            results = {
                "model": counts(predicted, batch["target"]),
                "copy_input": counts(batch["current"], batch["target"]),
            }
            for row in range(len(predicted)):
                path, source_view, views = dataset.items[index]
                record = {
                    "index": index,
                    "object_id": path.stem,
                    "source_view": source_view,
                    "target_view": evaluation_target(source_view, views),
                }
                for name, values in results.items():
                    totals[name].append(values[row].tolist())
                    record[name] = values[row].tolist()
                stream.write(json.dumps(record) + "\n")
                index += 1
            print(f"{index}/{len(pairs)} pairs", flush=True)

    summary = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": checkpoint["step"],
        "split": args.split,
        "objects": len({dataset.items[index][0] for index in pairs}),
        "view_pairs": len(pairs),
        "sampling_steps": args.steps,
        "seed": args.seed,
        "device": target_device.type,
        "counts_columns": COUNTS,
        "metrics": {name: metrics(rows) for name, rows in totals.items()},
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary["metrics"], indent=2))


if __name__ == "__main__":
    main()
