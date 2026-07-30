import json
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from hashlib import blake2b
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
    relative_sensor_action,
    sample_sensor_poses,
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

    output_directory.mkdir(parents=True)
    sensor_directions = make_ray_directions(config.sensor).astype(np.float32)
    np.save(output_directory / "ray_directions.npy", sensor_directions)
    metadata = {
        "config": {
            section: asdict(getattr(config, section))
            for section in ("sensor", "dataset")
        },
        "meshes": len(mesh_paths),
        "synsets": sorted({_synset(path) for path in mesh_paths}),
    }
    (output_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    items = split_meshes(mesh_paths, config)
    with (output_directory / "manifest.jsonl").open(
        "w", encoding="utf-8"
    ) as manifest:
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


class ViewDataset(Dataset[dict[str, np.ndarray]]):
    def __init__(
        self,
        directory: Path,
        split: str,
        cache_size: int = 16,
    ) -> None:
        rows = [
            json.loads(line)
            for line in (directory / "manifest.jsonl").read_text().splitlines()
        ]
        self.items = [
            (directory / row["sample"], source, row["views"])
            for row in rows
            if row["split"] == split
            for source in range(row["views"])
        ]
        self.ray_directions = np.load(directory / "ray_directions.npy")
        metadata = json.loads((directory / "metadata.json").read_text())
        sensor = metadata["config"]["sensor"]
        self.range_center = np.float32(sensor["distance"])
        self.range_scale = np.float32(1.0)
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
            sample = {
                name: file[name]
                for name in (
                    "scan_offsets",
                    "ray_indices",
                    "ranges",
                    "object_from_sensors",
                )
            }
        self.cache[path] = sample
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)
        return sample

    def scan_state(self, sample: dict[str, np.ndarray], view: int) -> np.ndarray:
        start, end = sample["scan_offsets"][view : view + 2]
        indices = sample["ray_indices"][start:end]
        ranges = sample["ranges"][start:end]
        height, width = self.ray_directions.shape[:2]
        state = np.zeros((2, height * width), dtype=np.float32)
        state[0] = -1.0
        state[0, indices] = 1.0
        state[1, indices] = (ranges - self.range_center) / self.range_scale
        return state.reshape(2, height, width)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        path, source_view, views = self.items[index]
        sample = self._load(path)
        if self.training:
            target_view = np.random.randint(views - 1)
            target_view += target_view >= source_view
        else:
            target_view = (source_view + 1 + views // 3) % views

        source_pose = sample["object_from_sensors"][source_view]
        target_pose = sample["object_from_sensors"][target_view]
        return {
            "current": self.scan_state(sample, source_view),
            "target": self.scan_state(sample, target_view),
            "action": relative_sensor_action(source_pose, target_pose),
        }
