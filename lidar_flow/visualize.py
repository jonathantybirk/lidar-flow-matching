import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


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

