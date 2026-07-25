from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lidar_flow.geometry import points_from_scan


def set_equal_axes(axis, points: np.ndarray) -> None:
    center = (points.min(axis=0) + points.max(axis=0)) / 2
    radius = np.ptp(points, axis=0).max() / 2 * 1.05
    axis.set_xlim(center[0] - radius, center[0] + radius)
    axis.set_ylim(center[1] - radius, center[1] + radius)
    axis.set_zlim(center[2] - radius, center[2] + radius)
    axis.set_box_aspect((1, 1, 1))
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    axis.set_zlabel("z (up)")
    axis.view_init(elev=18, azim=-65)


def save_inspection_figure(
    mesh_path: Path,
    sample_path: Path,
    ray_directions_path: Path,
    view_index: int,
    output_path: Path,
    show: bool = False,
) -> None:
    with np.load(sample_path) as sample:
        object_from_original = sample["object_from_original"]
        start, end = sample["scan_offsets"][view_index : view_index + 2]
        next_view = (view_index + 1 + len(sample["object_from_sensors"]) // 3) % len(
            sample["object_from_sensors"]
        )
        next_start, next_end = sample["scan_offsets"][next_view : next_view + 2]
        first_pose = sample["object_from_sensors"][view_index]
        second_pose = sample["object_from_sensors"][next_view]
        first_ranges = sample["ranges"][start:end]
        first_indices = sample["ray_indices"][start:end]
        second_ranges = sample["ranges"][next_start:next_end]
        second_indices = sample["ray_indices"][next_start:next_end]
    sensor_directions = np.load(ray_directions_path)
    first_points = points_from_scan(
        first_ranges, first_indices, sensor_directions, first_pose
    )
    second_points = points_from_scan(
        second_ranges, second_indices, sensor_directions, second_pose
    )

    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    mesh.apply_transform(object_from_original)

    mesh_vertices = mesh.vertices
    figure = plt.figure(figsize=(18, 5))
    mesh_axis = figure.add_subplot(131, projection="3d")
    first_axis = figure.add_subplot(132, projection="3d")
    second_axis = figure.add_subplot(133, projection="3d")

    mesh_axis.add_collection3d(
        Poly3DCollection(
            mesh_vertices[mesh.faces],
            facecolor="tab:blue",
            edgecolor="navy",
            linewidth=0.15,
            alpha=0.45,
        )
    )
    mesh_axis.set_title(f"Original mesh ({len(mesh.faces)} faces)")

    for axis, points, pose, title, color in (
        (first_axis, first_points, first_pose, "Current", "tab:orange"),
        (second_axis, second_points, second_pose, "Target", "tab:blue"),
    ):
        position = pose[:3, 3]
        forward = pose[:3, 2]
        axis.scatter(*points.T, s=9, color=color)
        axis.scatter(*position, s=90, color="tab:red")
        axis.quiver(
            *position,
            *forward,
            length=0.45,
            color="tab:red",
            arrow_length_ratio=0.2,
        )
        axis.set_title(f"{title} observation ({len(points)} returns)")

    shared_bounds = np.vstack(
        (
            mesh_vertices,
            first_points,
            second_points,
            first_pose[:3, 3],
            second_pose[:3, 3],
        )
    )
    for axis in (mesh_axis, first_axis, second_axis):
        set_equal_axes(axis, shared_bounds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
