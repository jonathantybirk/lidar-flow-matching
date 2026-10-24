import argparse
from concurrent.futures import ProcessPoolExecutor
from hashlib import blake2b
import json
import os
from pathlib import Path
from zipfile import ZipFile

from huggingface_hub import snapshot_download
import matplotlib.pyplot as plt
import numpy as np
import trimesh

from lidar_flow.dataset import evaluation_target
from lidar_flow.geometry import (
    clean_mesh,
    make_ray_directions,
    normalize_mesh,
    object_mesh,
    points_from_scan,
    sample_sensor_poses,
    simulate_scan,
)
from lidar_flow.visualize import draw_mesh, draw_sensor, set_equal_axes

ROOT = Path(__file__).parent
DATA = ROOT / "data"

SENSOR = {
    "height": 32,
    "width": 32,
    "horizontal_fov_degrees": 60.0,
    "vertical_fov_degrees": 60.0,
    "distance": 2.25,
    "minimum_elevation_degrees": -10.0,
    "maximum_elevation_degrees": 15.0,
}
DATASET = {
    "training_views": 32,
    "evaluation_views": 20,
    "validation_fraction": 0.1,
    "test_fraction": 0.1,
    "seed": 42,
}
RAY_DIRECTIONS = make_ray_directions(
    SENSOR["height"],
    SENSOR["width"],
    SENSOR["horizontal_fov_degrees"],
    SENSOR["vertical_fov_degrees"],
).astype(np.float32)


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


def download_synset(synset: str, limit: int | None) -> None:
    archives = DATA / "archives"
    snapshot_download(
        repo_id="ShapeNet/ShapeNetCore",
        repo_type="dataset",
        allow_patterns=[f"{synset}.zip"],
        local_dir=archives,
    )
    archive = archives / f"{synset}.zip"
    with ZipFile(archive) as zip_file:
        members = sorted(
            member
            for member in zip_file.namelist()
            if member.endswith("/models/model_normalized.obj")
        )
        zip_file.extractall(DATA / "raw" / synset, members[:limit])
    archive.unlink()


def synset_of(mesh_path: Path) -> str:
    return mesh_path.parents[2].name


def split_meshes(mesh_paths: list[Path]) -> list[tuple[Path, str]]:
    rng = np.random.default_rng(DATASET["seed"])
    items: list[tuple[Path, str]] = []
    for synset in sorted({synset_of(path) for path in mesh_paths}):
        category = [path for path in mesh_paths if synset_of(path) == synset]
        shuffled = rng.permutation(len(category))
        validation_count = round(len(category) * DATASET["validation_fraction"])
        test_count = round(len(category) * DATASET["test_fraction"])
        splits = np.full(len(category), "train", dtype=object)
        splits[shuffled[:validation_count]] = "validation"
        splits[shuffled[validation_count : validation_count + test_count]] = "test"
        items.extend(zip(category, splits.tolist(), strict=True))
    return items


def generate_sample(item: tuple[Path, str]) -> tuple[str, str, dict[str, np.ndarray]]:
    mesh_path, split = item
    sample_id = f"{synset_of(mesh_path)}_{mesh_path.parents[1].name}"
    digest = blake2b(f"{DATASET['seed']}:{sample_id}".encode(), digest_size=8)
    rng = np.random.default_rng(int.from_bytes(digest.digest()))
    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    normalized = normalize_mesh(clean_mesh(mesh))
    poses = sample_sensor_poses(
        DATASET["training_views" if split == "train" else "evaluation_views"],
        SENSOR["distance"],
        SENSOR["minimum_elevation_degrees"],
        SENSOR["maximum_elevation_degrees"],
        rng,
    )
    scans = [simulate_scan(normalized.mesh, pose, RAY_DIRECTIONS) for pose in poses]
    if any(len(ranges) == 0 for _, ranges in scans):
        raise ValueError(f"A sensor view missed mesh {sample_id}")
    counts = np.fromiter((len(ranges) for _, ranges in scans), dtype=np.int32)
    arrays = {
        "object_from_original": normalized.object_from_original,
        "scan_offsets": np.concatenate(([0], np.cumsum(counts))).astype(np.int32),
        "ray_indices": np.concatenate([indices for indices, _ in scans]),
        "ranges": np.concatenate([ranges for _, ranges in scans]),
        "object_from_sensors": poses.astype(np.float32),
    }
    return sample_id, split, arrays


def generate_dataset(mesh_paths: list[Path], output: Path, workers: int) -> None:
    if output.exists():
        raise FileExistsError(f"Dataset already exists: {output}")
    output.mkdir(parents=True)
    np.save(output / "ray_directions.npy", RAY_DIRECTIONS)
    metadata = {
        "config": {"sensor": SENSOR, "dataset": DATASET},
        "meshes": len(mesh_paths),
        "synsets": sorted({synset_of(path) for path in mesh_paths}),
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    items = split_meshes(mesh_paths)
    with (
        ProcessPoolExecutor(max_workers=workers) as executor,
        (output / "manifest.jsonl").open("w") as manifest,
    ):
        samples = executor.map(generate_sample, items, chunksize=1)
        for index, (sample_id, split, arrays) in enumerate(samples, start=1):
            np.savez(output / f"{sample_id}.npz", **arrays)
            row = {
                "model_id": sample_id,
                "synset": sample_id.split("_", 1)[0],
                "sample": f"{sample_id}.npz",
                "split": split,
                "views": len(arrays["object_from_sensors"]),
            }
            manifest.write(json.dumps(row) + "\n")
            if index % 100 == 0 or index == len(items):
                print(f"Generated {index}/{len(items)} meshes", flush=True)


def inspect_sample(
    dataset: Path, model_id: str, view: int, output: Path, show: bool
) -> None:
    directions = np.load(dataset / "ray_directions.npy")
    with np.load(dataset / f"{model_id}.npz") as sample:
        poses = sample["object_from_sensors"]
        offsets = sample["scan_offsets"]
        mesh = object_mesh(DATA / "raw", model_id, sample["object_from_original"])
        target_view = evaluation_target(view, len(poses))
        scans = {}
        for name, index in (("Current", view), ("Target", target_view)):
            start, end = offsets[index : index + 2]
            scans[name] = points_from_scan(
                sample["ranges"][start:end],
                sample["ray_indices"][start:end],
                directions,
                poses[index],
            )

    figure = plt.figure(figsize=(18, 5))
    axes = [figure.add_subplot(1, 3, index, projection="3d") for index in range(1, 4)]
    draw_mesh(axes[0], mesh)
    for axis, (title, points), index, color in zip(
        axes[1:], scans.items(), (view, target_view), ("tab:orange", "tab:blue")
    ):
        axis.scatter(*points.T, s=9, color=color)
        draw_sensor(axis, poses[index])
        axis.set_title(f"{title} observation ({len(points)} returns)")
    bounds = np.vstack(
        (mesh.vertices, *scans.values(), poses[view][:3, 3], poses[target_view][:3, 3])
    )
    for axis in axes:
        set_equal_axes(axis, bounds)

    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    if show:
        plt.show()
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.command == "download":
        for synset in args.synsets:
            download_synset(synset, args.limit)
    elif args.command == "generate":
        meshes = sorted(
            path
            for synset in args.synsets
            for path in (DATA / "raw" / synset).glob("**/models/model_normalized.obj")
        )[: args.limit]
        if not meshes:
            raise ValueError("No ShapeNet meshes found")
        generate_dataset(meshes, DATA / "processed" / args.dataset, args.workers)
    elif args.command == "inspect":
        inspect_sample(
            DATA / "processed" / args.dataset,
            f"{args.synset}_{args.model_id}",
            args.view,
            ROOT / "artifacts" / "inspection" / f"{args.model_id}_{args.view:04d}.png",
            args.show,
        )


if __name__ == "__main__":
    main()
