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
    target_points: int,
    output_path: Path,
    show: bool = False,
) -> None:
    with np.load(sample_path) as sample:
        complete_points = sample["surface_points"][:target_points]
        object_from_original = sample["object_from_original"]
        start, end = sample["scan_offsets"][view_index : view_index + 2]
        ranges = sample["ranges"][start:end]
        ray_indices = sample["ray_indices"][start:end]
        object_from_sensor = sample["object_from_sensors"][view_index]
    sensor_directions = np.load(ray_directions_path)
    partial_points = points_from_scan(
        ranges, ray_indices, sensor_directions, object_from_sensor
    )

    mesh = trimesh.load_mesh(mesh_path, force="mesh", process=False)
    mesh.apply_transform(object_from_original)

    mesh_vertices = mesh.vertices
    sensor_position = object_from_sensor[:3, 3]

    figure = plt.figure(figsize=(18, 5))
    mesh_axis = figure.add_subplot(131, projection="3d")
    complete_axis = figure.add_subplot(132, projection="3d")
    observation_axis = figure.add_subplot(133, projection="3d")

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

    complete_axis.scatter(*complete_points.T, s=6, color="tab:blue", alpha=0.9)
    complete_axis.set_title(f"Complete surface ({len(complete_points)} points)")

    observation_axis.scatter(
        *complete_points.T,
        s=5,
        color="lightgray",
        alpha=0.5,
        label="complete",
    )
    observation_axis.scatter(
        *partial_points.T,
        s=9,
        color="tab:orange",
        label=f"observed ({len(partial_points)})",
    )
    observation_axis.scatter(
        *sensor_position,
        s=90,
        color="tab:red",
        label="sensor",
    )

    forward = object_from_sensor[:3, 2]
    observation_axis.quiver(
        *sensor_position,
        *forward,
        length=0.45,
        color="tab:red",
        arrow_length_ratio=0.2,
    )
    observation_axis.set_title("LiDAR observation and sensor")

    object_bounds = np.vstack((mesh_vertices, complete_points, partial_points))
    shared_bounds = np.vstack(
        (object_bounds, sensor_position, sensor_position + forward * 0.45)
    )
    for axis in (mesh_axis, complete_axis, observation_axis):
        set_equal_axes(axis, shared_bounds)
    observation_axis.legend()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
