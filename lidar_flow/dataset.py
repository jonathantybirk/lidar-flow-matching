import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import blake2b
from collections import OrderedDict
from pathlib import Path
from typing import Iterator

import numpy as np
import trimesh
from torch.utils.data import Dataset

from lidar_flow.config import DataConfig
from lidar_flow.geometry import (
    clean_mesh,
    make_ray_directions,
    normalize_mesh,
    sample_sensor_poses,
    sample_surface,
    simulate_scan,
)


def discover_meshes(raw_directories: list[Path]) -> list[Path]:
    return sorted(
        path
        for directory in raw_directories
        for path in directory.glob("**/models/model_normalized.obj")
    )


def _synset(mesh_path: Path) -> str:
    return mesh_path.parents[2].name


def split_meshes(mesh_paths: list[Path], config: DataConfig) -> list[tuple[Path, str]]:
    rng = np.random.default_rng(config.dataset.seed)
    items: list[tuple[Path, str]] = []
    for synset in sorted({_synset(path) for path in mesh_paths}):
        category = [path for path in mesh_paths if _synset(path) == synset]
        shuffled = rng.permutation(len(category))
        validation_count = round(
            len(category) * config.dataset.validation_fraction
        )
        test_count = round(len(category) * config.dataset.test_fraction)
        splits = np.full(len(category), "train", dtype=object)
        splits[shuffled[:validation_count]] = "validation"
        splits[shuffled[validation_count : validation_count + test_count]] = "test"
        items.extend(zip(category, splits.tolist(), strict=True))
    return items


def _mesh_seed(seed: int, model_id: str) -> int:
    digest = blake2b(f"{seed}:{model_id}".encode(), digest_size=8).digest()
    return int.from_bytes(digest)


def _generate_sample(
    item: tuple[Path, str], config: DataConfig, sensor_directions: np.ndarray
) -> tuple[str, str, dict[str, np.ndarray]]:
    mesh_path, split = item
    model_id = mesh_path.parents[1].name
    synset = _synset(mesh_path)
    sample_id = f"{synset}_{model_id}"
    rng = np.random.default_rng(_mesh_seed(config.dataset.seed, sample_id))
    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    normalized = normalize_mesh(clean_mesh(mesh))
    surface_points = sample_surface(
        normalized.mesh,
        config.surface.candidate_points,
        config.surface.target_points,
        config.surface.normal_weight,
        rng,
    ).astype(np.float32)
    view_count = (
        config.dataset.training_views
        if split == "train"
        else config.dataset.evaluation_views
    )
    poses = sample_sensor_poses(
        view_count,
        config.sensor.distance,
        config.sensor.minimum_elevation_degrees,
        config.sensor.maximum_elevation_degrees,
        rng,
    )
    scans = [simulate_scan(normalized.mesh, pose, sensor_directions) for pose in poses]
    if any(len(ranges) == 0 for _, ranges in scans):
        raise ValueError(f"A sensor view missed mesh {sample_id}")
    counts = np.fromiter((len(ranges) for _, ranges in scans), dtype=np.int32)
    arrays = {
        "surface_points": surface_points,
        "object_from_original": normalized.object_from_original,
        "scan_offsets": np.concatenate(([0], np.cumsum(counts))).astype(np.int32),
        "ray_indices": np.concatenate([indices for indices, _ in scans]),
        "ranges": np.concatenate([ranges for _, ranges in scans]),
        "object_from_sensors": poses.astype(np.float32),
    }
    return sample_id, split, arrays


def _samples(
    items: list[tuple[Path, str]],
    config: DataConfig,
    sensor_directions: np.ndarray,
    workers: int,
) -> Iterator[tuple[str, str, dict[str, np.ndarray]]]:
    if workers == 1:
        for item in items:
            yield _generate_sample(item, config, sensor_directions)
        return
    with ProcessPoolExecutor(max_workers=workers) as executor:
        yield from executor.map(
            _generate_sample,
            items,
            [config] * len(items),
            [sensor_directions] * len(items),
            chunksize=1,
        )


def generate_dataset(
    mesh_paths: list[Path], output_directory: Path, config: DataConfig, workers: int
) -> None:
    if output_directory.exists():
        raise FileExistsError(f"Dataset already exists: {output_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)
    sensor_directions = make_ray_directions(config.sensor).astype(np.float32)
    np.save(output_directory / "ray_directions.npy", sensor_directions)
    metadata = {
        "config": {
            section: asdict(getattr(config, section))
            for section in ("surface", "sensor", "dataset")
        },
        "meshes": len(mesh_paths),
        "synsets": sorted({_synset(path) for path in mesh_paths}),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    items = split_meshes(mesh_paths, config)
    manifest_path = output_directory / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for index, (model_id, split, arrays) in enumerate(
            _samples(items, config, sensor_directions, workers), start=1
        ):
            sample_path = output_directory / f"{model_id}.npz"
            np.savez(sample_path, **arrays)
            manifest.write(
                json.dumps(
                    {
                        "model_id": model_id,
                        "synset": model_id.split("_", 1)[0],
                        "sample": sample_path.name,
                        "split": split,
                        "views": len(arrays["object_from_sensors"]),
                    }
                )
                + "\n"
            )
            if index % 100 == 0 or index == len(items):
                print(f"Generated {index}/{len(items)} meshes", flush=True)


class CompletionDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(
        self,
        directory: Path,
        split: str,
        condition_points: int,
        target_points: int,
        cache_size: int = 16,
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("Training code exposes only train and validation splits")
        rows = [
            json.loads(line)
            for line in (directory / "manifest.jsonl").read_text().splitlines()
        ]
        self.items = [
            (directory / row["sample"], view)
            for row in rows
            if row["split"] == split
            for view in range(row["views"])
        ]
        self.ray_directions = np.load(directory / "ray_directions.npy").reshape(-1, 3)
        self.condition_points = condition_points
        self.target_points = target_points
        self.training = split == "train"
        self.cache_size = cache_size
        self.cache: OrderedDict[Path, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self) -> int:
        return len(self.items)

    def _load(self, path: Path) -> dict[str, np.ndarray]:
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        with np.load(path) as file:
            sample = {name: file[name] for name in file.files}
        self.cache[path] = sample
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return sample

    def _indices(self, count: int, size: int) -> np.ndarray:
        if count <= size:
            return np.arange(count)
        if self.training:
            return np.random.choice(count, size=size, replace=False)
        return np.linspace(0, count - 1, size, dtype=np.int64)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        path, view = self.items[index]
        sample = self._load(path)
        start, end = sample["scan_offsets"][view : view + 2]
        ranges = sample["ranges"][start:end]
        ray_indices = sample["ray_indices"][start:end]
        pose = sample["object_from_sensors"][view]
        directions = self.ray_directions[ray_indices] @ pose[:3, :3].T
        points = pose[:3, 3] + directions * ranges[:, None]
        sensor_position = pose[:3, 3]
        sensor_forward = pose[:3, 2]
        surface = sample["surface_points"]

        if self.training:
            yaw = np.random.uniform(0.0, 2.0 * np.pi)
            cosine, sine = np.cos(yaw), np.sin(yaw)
            rotation = np.array(
                ((cosine, -sine, 0.0), (sine, cosine, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float32,
            )
            points = points @ rotation.T
            directions = directions @ rotation.T
            sensor_position = sensor_position @ rotation.T
            sensor_forward = sensor_forward @ rotation.T
            surface = surface @ rotation.T

        selected = self._indices(len(points), self.condition_points)
        observed_features = np.zeros((self.condition_points, 10), dtype=np.float32)
        observed_xyz = np.zeros((self.condition_points, 3), dtype=np.float32)
        observed_mask = np.zeros(self.condition_points, dtype=np.bool_)
        count = len(selected)
        observed_xyz[:count] = points[selected]
        observed_features[:count] = np.concatenate(
            (
                points[selected],
                directions[selected],
                ranges[selected, None],
                points[selected] - sensor_position,
            ),
            axis=1,
        )
        observed_mask[:count] = True

        target_indices = self._indices(len(surface), self.target_points)
        sensor_features = np.concatenate(
            (
                sensor_position,
                sensor_forward,
                np.array(
                    [
                        len(points) / len(self.ray_directions),
                        ranges.mean(),
                        ranges.std(),
                    ],
                    dtype=np.float32,
                ),
            )
        ).astype(np.float32)
        return {
            "observed_features": observed_features,
            "observed_xyz": observed_xyz,
            "observed_mask": observed_mask,
            "sensor_features": sensor_features,
            "target": surface[target_indices].astype(np.float32),
        }
