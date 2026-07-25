import torch
from torch import Tensor

from lidar_flow.model import LidarFlowTransformer


def flow_loss(
    model: LidarFlowTransformer,
    batch: dict[str, Tensor],
    origin: Tensor | None = None,
    time: Tensor | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    target = batch["target"]
    origin = torch.randn_like(target) if origin is None else origin
    time = torch.rand(len(target), device=target.device) if time is None else time
    state = torch.lerp(origin, target, time[:, None, None, None])
    velocity = model(
        state,
        time,
        batch["current"],
        batch["action"],
    )
    error = (velocity.float() - (target - origin).float()).square()

    hits = target[:, 0] > 0
    occupancy = error[:, 0].mean()
    ranges = (error[:, 1][hits].mean() + error[:, 1][~hits].mean()) / 2
    return (occupancy + ranges) / 2, {
        "occupancy": occupancy,
        "range": ranges,
    }


@torch.no_grad()
def sample(
    model: LidarFlowTransformer,
    batch: dict[str, Tensor],
    steps: int = 8,
    origin: Tensor | None = None,
) -> Tensor:
    if origin is None:
        origin = torch.randn_like(batch["current"])
    state = origin.clone()
    context = model.encode(batch["current"])
    step_size = 1 / steps
    for step in range(steps):
        time = torch.full(
            (len(state),), step / steps, device=state.device, dtype=state.dtype
        )
        first = model.velocity(state, time, batch["action"], context)
        prediction = state + step_size * first
        if step == steps - 1:
            state = prediction
        else:
            second = model.velocity(
                prediction,
                time + step_size,
                batch["action"],
                context,
            )
            state = state + step_size * (first + second) / 2
    return state
