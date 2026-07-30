import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from pathlib import Path
import threading
import webbrowser

import numpy as np
import torch

from lidar_flow.config import ModelConfig
from lidar_flow.dataset import ViewDataset
from lidar_flow.flow import sample
from lidar_flow.geometry import (
    orbit_pose,
    object_mesh,
    points_from_scan,
    points_from_state,
    relative_sensor_action,
    simulate_scan,
)
from lidar_flow.model import LidarFlowTransformer, device

ROOT = Path(__file__).parent
DEFAULT_CHECKPOINT = ROOT / "artifacts/checkpoints/a100-qknorm-best.pt"
PAGE = ROOT / "demo.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Explore action-conditioned LiDAR generation"
    )
    parser.add_argument(
        "checkpoint",
        type=Path,
        nargs="?",
        default=DEFAULT_CHECKPOINT,
        help=f"model weights (default: {DEFAULT_CHECKPOINT.relative_to(ROOT)})",
    )
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/vehicles")
    parser.add_argument("--raw", type=Path, default=ROOT / "data/raw")
    parser.add_argument(
        "--model-id", help="ShapeNet sample ID; defaults to the first sample"
    )
    parser.add_argument("--view", type=int, default=0, help="fixed input view")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args()


def choose_sample(data: Path, model_id: str | None) -> tuple[str, str]:
    rows = [
        json.loads(line)
        for line in (data / "manifest.jsonl").read_text().splitlines()
    ]
    row = rows[0] if model_id is None else next(
        (row for row in rows if row["model_id"] == model_id), None
    )
    if row is None:
        raise ValueError(f"Unknown model ID: {model_id}")
    return row["model_id"], row["split"]


def compact_mesh(mesh, limit: int = 4000) -> tuple[np.ndarray, np.ndarray]:
    faces = mesh.faces
    if len(faces) > limit:
        faces = faces[np.linspace(0, len(faces) - 1, limit, dtype=int)]
    indices, inverse = np.unique(faces, return_inverse=True)
    return mesh.vertices[indices], inverse.reshape(-1, 3)


class Demo:
    def __init__(self, args: argparse.Namespace) -> None:
        model_id, split = choose_sample(args.data, args.model_id)
        dataset = ViewDataset(args.data, split)
        matches = [
            index
            for index, (path, view, _) in enumerate(dataset.items)
            if path.stem == model_id and view == args.view
        ]
        if not matches:
            raise ValueError(f"View {args.view} does not exist for {model_id}")
        current = dataset[matches[0]]["current"]

        target_device = device()
        checkpoint = torch.load(
            args.checkpoint, map_location=target_device, weights_only=True
        )
        model = LidarFlowTransformer(
            ModelConfig(**checkpoint["config"]["model"]), dataset.ray_directions
        ).to(target_device)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        with np.load(args.data / f"{model_id}.npz") as stored:
            poses = stored["object_from_sensors"]
            mesh = object_mesh(args.raw, model_id, stored["object_from_original"])
        source_pose = poses[args.view]
        source = points_from_state(
            current,
            dataset.ray_directions,
            dataset.range_center,
            dataset.range_scale,
        ) @ source_pose[:3, :3].T + source_pose[:3, 3]
        vertices, faces = compact_mesh(mesh)

        self.dataset = dataset
        self.mesh = mesh
        self.model = model
        self.source_pose = source_pose
        self.current = torch.from_numpy(current[None]).to(target_device)
        self.context = model.encode(self.current)
        self.steps = args.steps
        self.device = target_device
        self.generator = torch.Generator(device=target_device).manual_seed(args.seed)
        self.noise = self._noise()
        self.lock = threading.Lock()
        self.scene = {
            "model": model_id,
            "mesh": vertices.tolist(),
            "faces": faces.tolist(),
            "source": source.tolist(),
            "sourcePosition": source_pose[:3, 3].tolist(),
            "distance": float(np.linalg.norm(source_pose[:3, 3])),
        }

    def _noise(self) -> torch.Tensor:
        return torch.randn(
            self.current.shape,
            generator=self.generator,
            device=self.device,
        )

    @torch.no_grad()
    def predict(self, azimuth: float, elevation: float, resample: bool) -> dict:
        with self.lock:
            if resample:
                self.noise = self._noise()
            pose = orbit_pose(
                np.deg2rad(azimuth),
                np.deg2rad(elevation),
                self.scene["distance"],
            )
            action = torch.from_numpy(
                relative_sensor_action(self.source_pose, pose)[None]
            ).to(self.device)
            generated = sample(
                self.model,
                {"current": self.current, "action": action},
                steps=self.steps,
                origin=self.noise,
                context=self.context,
            )[0].cpu().numpy()

        indices, ranges = simulate_scan(
            self.mesh, pose, self.dataset.ray_directions
        )
        predicted = points_from_state(
            generated,
            self.dataset.ray_directions,
            self.dataset.range_center,
            self.dataset.range_scale,
        ) @ pose[:3, :3].T + pose[:3, 3]
        actual = points_from_scan(
            ranges, indices, self.dataset.ray_directions, pose
        )

        predicted_hits = generated[0].reshape(-1) > 0
        actual_hits = np.zeros(predicted_hits.shape, dtype=bool)
        actual_hits[indices] = True
        intersection = predicted_hits & actual_hits
        union = predicted_hits | actual_hits
        actual_ranges = np.zeros(predicted_hits.shape, dtype=np.float32)
        actual_ranges[indices] = ranges - self.dataset.range_center
        rmse = np.sqrt(
            np.mean(
                (generated[1].reshape(-1)[intersection] - actual_ranges[intersection])
                ** 2
            )
        ) if intersection.any() else float("nan")
        return {
            "predicted": predicted.tolist(),
            "actual": actual.tolist(),
            "targetPosition": pose[:3, 3].tolist(),
            "iou": float(intersection.sum() / union.sum()),
            "rmse": float(rmse),
        }


def handler(demo: Demo):
    class Handler(BaseHTTPRequestHandler):
        def send(self, content: bytes, content_type: str) -> None:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def do_GET(self) -> None:
            if self.path == "/":
                self.send(PAGE.read_bytes(), "text/html; charset=utf-8")
            elif self.path == "/scene":
                self.send(json.dumps(demo.scene).encode(), "application/json")
            else:
                self.send_error(404)

        def do_POST(self) -> None:
            if self.path != "/predict":
                self.send_error(404)
                return
            length = int(self.headers.get("Content-Length", 0))
            request = json.loads(self.rfile.read(length))
            result = demo.predict(
                float(request["azimuth"]),
                float(request["elevation"]),
                bool(request.get("resample")),
            )
            self.send(json.dumps(result).encode(), "application/json")

        def log_message(self, format: str, *args) -> None:
            pass

    return Handler


def main() -> None:
    args = parse_args()
    demo = Demo(args)
    server = HTTPServer(("127.0.0.1", args.port), handler(demo))
    url = f"http://127.0.0.1:{args.port}"
    print(f"LiDAR WAM demo: {url}\nPress Ctrl+C to stop.", flush=True)
    if not args.no_open:
        threading.Timer(0.25, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
