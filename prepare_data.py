import argparse
import os
from pathlib import Path

from lidar_flow.config import load_data_config
from lidar_flow.dataset import discover_meshes, generate_dataset
from lidar_flow.download import download_synset
from lidar_flow.visualize import save_inspection_figure

ROOT = Path(__file__).parent
DATA = ROOT / "data"
CONFIG = ROOT / "configs" / "data.toml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ShapeNet LiDAR data")
    commands = parser.add_subparsers(dest="command", required=True)

    download = commands.add_parser("download", help="Download a ShapeNet synset")
    download.add_argument("synsets", nargs="+")
    download.add_argument("--limit", type=int)

    generate = commands.add_parser("generate", help="Generate LiDAR observations")
    generate.add_argument("dataset")
    generate.add_argument("synsets", nargs="+")
    generate.add_argument("--limit", type=int)
    generate.add_argument(
        "--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2)
    )

    inspect = commands.add_parser("inspect", help="Visualize one generated example")
    inspect.add_argument("dataset")
    inspect.add_argument("synset")
    inspect.add_argument("model_id")
    inspect.add_argument("--view", type=int, default=0)
    inspect.add_argument("--show", action="store_true", help="Open a rotatable 3D view")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "download":
        for synset in args.synsets:
            download_synset(synset, DATA, args.limit)

    elif args.command == "generate":
        meshes = discover_meshes(
            [DATA / "raw" / synset for synset in args.synsets]
        )
        if args.limit is not None:
            meshes = meshes[: args.limit]
        if not meshes:
            raise ValueError("No ShapeNet meshes found")

        generate_dataset(
            meshes,
            DATA / "processed" / args.dataset,
            load_data_config(CONFIG),
            args.workers,
        )

    elif args.command == "inspect":
        processed_directory = DATA / "processed" / args.dataset
        save_inspection_figure(
            DATA
            / "raw"
            / args.synset
            / args.synset
            / args.model_id
            / "models"
            / "model_normalized.obj",
            processed_directory / f"{args.synset}_{args.model_id}.npz",
            processed_directory / "ray_directions.npy",
            args.view,
            load_data_config(CONFIG).surface.target_points,
            ROOT / "artifacts" / "inspection" / f"{args.model_id}_{args.view:04d}.png",
            show=args.show,
        )


if __name__ == "__main__":
    main()
