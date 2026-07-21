import torch
from torch import Tensor

from lidar_flow.model import PointFlowTransformer


def nearest_neighbor_target(origin: Tensor, target: Tensor) -> Tensor:
    with torch.no_grad():
        indices = torch.cdist(origin.float(), target.float()).argmin(dim=-1)
        return target.gather(1, indices[:, :, None].expand(-1, -1, 3))


def chamfer_distance(first: Tensor, second: Tensor) -> Tensor:
    distances = torch.cdist(first.float(), second.float()).square()
    return distances.amin(dim=2).mean() + distances.amin(dim=1).mean()


def completion_loss(
    model: PointFlowTransformer,
    batch: dict[str, Tensor],
    chamfer_weight: float,
    observation_weight: float,
    origin: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    target = batch["target"]
    origin = torch.randn_like(target) if origin is None else origin
    time = torch.rand(len(target), device=target.device) if time is None else time
    matched_target = nearest_neighbor_target(origin, target)
    state = torch.lerp(origin, matched_target, time[:, None, None])
    target_velocity = matched_target - origin
    velocity = model(
        state,
        time,
        origin,
        batch["observed_features"],
        batch["observed_xyz"],
        batch["observed_mask"],
        batch["sensor_features"],
    )
    flow = (velocity.float() - target_velocity.float()).square().mean()
    endpoint = state.float() + (1 - time[:, None, None]) * velocity.float()
    chamfer = chamfer_distance(endpoint, target)

    observed_distances = torch.cdist(batch["observed_xyz"].float(), endpoint).square()
    observed = observed_distances.amin(dim=2)
    mask = batch["observed_mask"]
    observation = (observed * mask).sum() / mask.sum()
    total = flow + chamfer_weight * chamfer + observation_weight * observation
    return total, {"flow": flow, "chamfer": chamfer, "observation": observation}


@torch.no_grad()
def sample(
    model: PointFlowTransformer,
    batch: dict[str, Tensor],
    steps: int = 8,
    origin: Tensor | None = None,
) -> Tensor:
    if origin is None:
        origin = torch.randn(
            len(batch["observed_features"]),
            model.config.points,
            3,
            device=batch["observed_features"].device,
            dtype=batch["observed_features"].dtype,
        )
    state = origin.clone()
    condition, condition_mask = model.encode_condition(
        batch["observed_features"],
        batch["observed_xyz"],
        batch["observed_mask"],
        batch["sensor_features"],
    )
    step_size = 1 / steps
    for step in range(steps):
        time = torch.full(
            (len(state),), step / steps, device=state.device, dtype=state.dtype
        )
        first = model.velocity(state, time, origin, condition, condition_mask)
        prediction = state + step_size * first
        if step == steps - 1:
            state = prediction
        else:
            second = model.velocity(
                prediction, time + step_size, origin, condition, condition_mask
            )
            state = state + step_size * (first + second) / 2
    return state
