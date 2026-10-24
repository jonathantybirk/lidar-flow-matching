from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import trimesh

FloatArray = npt.NDArray[np.floating]


@dataclass(frozen=True)
class NormalizedMesh:
    mesh: trimesh.Trimesh
    object_from_original: FloatArray


def object_mesh(raw: Path, model_id: str, object_from_original: FloatArray):
    synset, model = model_id.split("_", 1)
    path = raw / synset / synset / model / "models" / "model_normalized.obj"
    mesh = clean_mesh(trimesh.load_mesh(path, force="mesh", process=False))
    mesh.apply_transform(object_from_original)
    return mesh


def surface_centroid(mesh: trimesh.Trimesh) -> FloatArray:
    if not np.isfinite(mesh.area) or mesh.area <= 0:
        raise ValueError("Mesh must have positive finite surface area")
    face_centroids = mesh.triangles.mean(axis=1)
    return np.average(face_centroids, axis=0, weights=mesh.area_faces)


def clean_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    cleaned = mesh.copy()
    cleaned.merge_vertices(merge_tex=True, merge_norm=True)
    cleaned.update_faces(cleaned.nondegenerate_faces())
    cleaned.update_faces(cleaned.unique_faces())
    cleaned.remove_unreferenced_vertices()
    return cleaned


def normalize_mesh(mesh: trimesh.Trimesh) -> NormalizedMesh:
    centroid = surface_centroid(mesh)
    centered_vertices = mesh.vertices - centroid
    radius = np.linalg.norm(centered_vertices, axis=1).max()
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Mesh must have nonzero finite radius")
    z_up_from_y_up = np.array(
        ((1.0, 0.0, 0.0), (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
        dtype=np.float32,
    )

    normalized = mesh.copy()
    normalized.vertices = centered_vertices @ z_up_from_y_up.T / radius

    object_from_original = np.eye(4, dtype=np.float32)
    object_from_original[:3, :3] = z_up_from_y_up / radius
    object_from_original[:3, 3] = -(z_up_from_y_up @ centroid) / radius
    return NormalizedMesh(normalized, object_from_original)


def look_at_origin(sensor_position: FloatArray) -> FloatArray:
    forward = -sensor_position / np.linalg.norm(sensor_position)
    reference_up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(forward, reference_up)) > 0.99:
        reference_up = np.array([0.0, 1.0, 0.0])

    right = np.cross(reference_up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)

    object_from_sensor = np.eye(4, dtype=np.float32)
    object_from_sensor[:3, :3] = np.column_stack((right, up, forward))
    object_from_sensor[:3, 3] = sensor_position
    return object_from_sensor


def orbit_pose(azimuth: float, elevation: float, distance: float) -> FloatArray:
    """Return an object-from-sensor pose on an orbit, with angles in radians."""
    position = distance * np.array(
        (
            np.cos(elevation) * np.cos(azimuth),
            np.cos(elevation) * np.sin(azimuth),
            np.sin(elevation),
        )
    )
    return look_at_origin(position)


def relative_sensor_action(source_pose: FloatArray, target_pose: FloatArray) -> FloatArray:
    """Encode a target sensor pose in the source sensor frame."""
    source_rotation = source_pose[:3, :3]
    rotation = source_rotation.T @ target_pose[:3, :3]
    translation = source_rotation.T @ (target_pose[:3, 3] - source_pose[:3, 3])
    return np.concatenate((rotation.reshape(-1), translation)).astype(np.float32)


def sample_sensor_poses(
    count: int,
    distance: float,
    minimum_elevation_degrees: float,
    maximum_elevation_degrees: float,
    rng: np.random.Generator,
) -> FloatArray:
    azimuth = 2 * np.pi * (np.arange(count) + rng.random(count)) / count
    elevation_bins = (np.arange(count) + rng.random(count)) / count
    rng.shuffle(elevation_bins)
    minimum_sine = np.sin(np.deg2rad(minimum_elevation_degrees))
    maximum_sine = np.sin(np.deg2rad(maximum_elevation_degrees))
    elevation = np.arcsin(minimum_sine + elevation_bins * (maximum_sine - minimum_sine))
    return np.stack(
        [orbit_pose(*angles, distance) for angles in zip(azimuth, elevation)]
    )


def make_ray_directions(
    height: int,
    width: int,
    horizontal_fov_degrees: float,
    vertical_fov_degrees: float,
) -> FloatArray:
    if height * width > np.iinfo(np.uint16).max:
        raise ValueError("Sensor grid exceeds the ray-index data type")
    horizontal = np.linspace(
        -np.deg2rad(horizontal_fov_degrees) / 2,
        np.deg2rad(horizontal_fov_degrees) / 2,
        width,
    )
    vertical = np.linspace(
        -np.deg2rad(vertical_fov_degrees) / 2,
        np.deg2rad(vertical_fov_degrees) / 2,
        height,
    )
    horizontal_grid, vertical_grid = np.meshgrid(horizontal, vertical)
    directions = np.stack(
        (
            np.tan(horizontal_grid),
            np.tan(vertical_grid),
            np.ones_like(horizontal_grid),
        ),
        axis=-1,
    )
    return directions / np.linalg.norm(directions, axis=-1, keepdims=True)


def simulate_scan(
    mesh: trimesh.Trimesh,
    object_from_sensor: FloatArray,
    sensor_directions: FloatArray,
) -> tuple[npt.NDArray[np.uint16], FloatArray]:
    object_directions = sensor_directions @ object_from_sensor[:3, :3].T
    flat_directions = object_directions.reshape(-1, 3)
    origins = np.repeat(object_from_sensor[None, :3, 3], len(flat_directions), axis=0)

    locations, ray_indices, _ = mesh.ray.intersects_location(
        ray_origins=origins,
        ray_directions=flat_directions,
        multiple_hits=False,
    )
    distances = np.linalg.norm(locations - origins[ray_indices], axis=1)
    order = np.argsort(ray_indices)
    return ray_indices[order].astype(np.uint16), distances[order].astype(np.float32)


def points_from_scan(
    ranges: FloatArray,
    ray_indices: npt.NDArray[np.integer],
    sensor_directions: FloatArray,
    object_from_sensor: FloatArray,
) -> FloatArray:
    sensor_hits = sensor_directions.reshape(-1, 3)[ray_indices]
    object_directions = sensor_hits @ object_from_sensor[:3, :3].T
    return object_from_sensor[:3, 3] + object_directions * ranges[:, None]


def points_from_state(
    state: FloatArray,
    sensor_directions: FloatArray,
    range_center: float,
) -> FloatArray:
    hits = state[0].reshape(-1) > 0
    ranges = range_center + np.clip(
        state[1].reshape(-1)[hits], -1.0, 1.0
    )
    return sensor_directions.reshape(-1, 3)[hits] * ranges[:, None]
