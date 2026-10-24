import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from lidar_flow.geometry import relative_sensor_action


def evaluation_target(source_view: int, views: int) -> int:
    """Fixed query view for validation and test pairs."""
    return (source_view + 1 + views // 3) % views


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
        state[1, indices] = ranges - self.range_center
        return state.reshape(2, height, width)

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        path, source_view, views = self.items[index]
        sample = self._load(path)
        if self.training:
            target_view = np.random.randint(views - 1)
            target_view += target_view >= source_view
        else:
            target_view = evaluation_target(source_view, views)

        source_pose = sample["object_from_sensors"][source_view]
        target_pose = sample["object_from_sensors"][target_view]
        return {
            "current": self.scan_state(sample, source_view),
            "target": self.scan_state(sample, target_view),
            "action": relative_sensor_action(source_pose, target_pose),
        }
