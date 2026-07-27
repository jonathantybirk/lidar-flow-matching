from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import trimesh
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lidar_flow.geometry import clean_mesh, points_from_scan


def object_mesh(raw: Path, model_id: str, object_from_original: np.ndarray):
    synset, model = model_id.split("_", 1)
    path = raw / synset / synset / model / "models" / "model_normalized.obj"
    mesh = clean_mesh(trimesh.load_mesh(path, force="mesh", process=False))
    mesh.apply_transform(object_from_original)
    return mesh


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


def draw_mesh(axis, mesh) -> None:
    axis.add_collection3d(
        Poly3DCollection(
            mesh.vertices[mesh.faces],
            facecolor="tab:blue",
            edgecolor="navy",
            linewidth=0.15,
            alpha=0.45,
        )
    )
    axis.set_title(f"Mesh ({len(mesh.faces)} faces)")


def draw_sensor(axis, pose: np.ndarray) -> None:
    axis.scatter(*pose[:3, 3], s=70, color="tab:red", depthshade=False)
    axis.quiver(
        *pose[:3, 3],
        *pose[:3, 2],
        length=0.45,
        color="tab:red",
        arrow_length_ratio=0.22,
    )


def save_inspection_figure(
    raw: Path,
    dataset_directory: Path,
    model_id: str,
    view_index: int,
    output_path: Path,
    show: bool = False,
) -> None:
    directions = np.load(dataset_directory / "ray_directions.npy")
    with np.load(dataset_directory / f"{model_id}.npz") as sample:
        poses = sample["object_from_sensors"]
        offsets = sample["scan_offsets"]
        target_view = (view_index + 1 + len(poses) // 3) % len(poses)
        mesh = object_mesh(raw, model_id, sample["object_from_original"])

        def scan(view: int) -> np.ndarray:
            start, end = offsets[view : view + 2]
            return points_from_scan(
                sample["ranges"][start:end],
                sample["ray_indices"][start:end],
                directions,
                poses[view],
            )

        current = scan(view_index)
        target = scan(target_view)

    figure = plt.figure(figsize=(18, 5))
    axes = [figure.add_subplot(1, 3, index, projection="3d") for index in range(1, 4)]
    draw_mesh(axes[0], mesh)
    for axis, points, view, title, color in (
        (axes[1], current, view_index, "Current", "tab:orange"),
        (axes[2], target, target_view, "Target", "tab:blue"),
    ):
        axis.scatter(*points.T, s=9, color=color)
        draw_sensor(axis, poses[view])
        axis.set_title(f"{title} observation ({len(points)} returns)")

    bounds = np.vstack(
        (
            mesh.vertices,
            current,
            target,
            poses[view_index][:3, 3],
            poses[target_view][:3, 3],
        )
    )
    for axis in axes:
        set_equal_axes(axis, bounds)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    if show:
        plt.show()
    plt.close(figure)
