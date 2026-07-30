import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from lidar_flow.config import ModelConfig


def device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class TimestepEmbedding(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.SiLU(),
            nn.Linear(4 * width, width),
        )

    def forward(self, time: Tensor) -> Tensor:
        half = self.width // 2
        frequencies = torch.exp(
            -math.log(10_000)
            * torch.arange(half, device=time.device, dtype=time.dtype)
            / half
        )
        angles = time[:, None] * frequencies[None]
        return self.mlp(torch.cat((angles.cos(), angles.sin()), dim=-1))


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return value * (1 + scale[:, None]) + shift[:, None]


class Attention(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.query_norm = nn.RMSNorm(self.head_width)
        self.key_norm = nn.RMSNorm(self.head_width)
        self.output = nn.Linear(width, width)

    def _heads(self, value: Tensor) -> Tensor:
        return value.unflatten(-1, (self.heads, self.head_width)).transpose(1, 2)

    def forward(self, query: Tensor, context: Tensor) -> Tensor:
        queries = self.query_norm(self._heads(self.query(query)))
        keys = self.key_norm(self._heads(self.key(context)))
        values = self._heads(self.value(context))
        attended = F.scaled_dot_product_attention(queries, keys, values)
        return self.output(attended.transpose(1, 2).flatten(2))


class FlowBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.self_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.cross_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.mlp_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.self_attention = Attention(width, heads)
        self.cross_attention = Attention(width, heads)
        self.mlp = nn.Sequential(
            nn.Linear(width, 4 * width),
            nn.GELU(),
            nn.Linear(4 * width, width),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 9 * width))
        nn.init.zeros_(self.modulation[-1].weight)
        nn.init.zeros_(self.modulation[-1].bias)

    def forward(
        self, query: Tensor, context: Tensor, conditioning: Tensor
    ) -> Tensor:
        values = self.modulation(conditioning).chunk(9, dim=-1)
        shift_self, scale_self, gate_self = values[:3]
        shift_cross, scale_cross, gate_cross = values[3:6]
        shift_mlp, scale_mlp, gate_mlp = values[6:]

        value = _modulate(self.self_norm(query), shift_self, scale_self)
        update = self.self_attention(value, value)
        query = query + gate_self[:, None] * update

        value = _modulate(self.cross_norm(query), shift_cross, scale_cross)
        update = self.cross_attention(value, context)
        query = query + gate_cross[:, None] * update

        update = self.mlp(_modulate(self.mlp_norm(query), shift_mlp, scale_mlp))
        return query + gate_mlp[:, None] * update


class LidarFlowTransformer(nn.Module):
    def __init__(self, config: ModelConfig, ray_directions: Tensor) -> None:
        super().__init__()
        self.config = config
        self.register_buffer(
            "ray_directions",
            torch.as_tensor(ray_directions, dtype=torch.float32),
        )
        self.grid_height, self.grid_width = ray_directions.shape[:2]
        patch = config.patch_size
        self.context_embedding = nn.Conv2d(
            5, config.width, kernel_size=patch, stride=patch
        )
        self.query_embedding = nn.Conv2d(
            8, config.width, kernel_size=patch, stride=patch
        )
        self.context_blocks = nn.ModuleList(
            [
                nn.TransformerEncoderLayer(
                    config.width,
                    config.heads,
                    dim_feedforward=4 * config.width,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(config.condition_layers)
            ]
        )
        self.context_norm = nn.LayerNorm(config.width)
        self.time_embedding = TimestepEmbedding(config.width)
        self.action_embedding = nn.Sequential(
            nn.Linear(12, 4 * config.width),
            nn.SiLU(),
            nn.Linear(4 * config.width, config.width),
        )
        self.conditioning_norm = nn.LayerNorm(config.width)
        self.flow_blocks = nn.ModuleList(
            [FlowBlock(config.width, config.heads) for _ in range(config.flow_layers)]
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, 2 * patch * patch)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @staticmethod
    def _tokens(features: Tensor, embedding: nn.Conv2d) -> Tensor:
        return embedding(features).flatten(2).transpose(1, 2)

    def encode(self, current: Tensor) -> Tensor:
        batch = len(current)
        directions = (
            self.ray_directions.permute(2, 0, 1)[None].expand(batch, -1, -1, -1)
        )
        context = self._tokens(
            torch.cat((current, directions), dim=1), self.context_embedding
        )
        for block in self.context_blocks:
            context = block(context)
        return self.context_norm(context)

    def query_geometry(self, action: Tensor) -> Tensor:
        rotation = action[:, :9].reshape(-1, 3, 3)
        directions = torch.einsum(
            "hwj,bkj->bkhw", self.ray_directions, rotation
        )
        position = action[:, 9:, None, None].expand(
            -1, -1, self.grid_height, self.grid_width
        )
        return torch.cat((position, directions), dim=1)

    def velocity(
        self,
        state: Tensor,
        time: Tensor,
        action: Tensor,
        context: Tensor,
    ) -> Tensor:
        query = self._tokens(
            torch.cat((state, self.query_geometry(action)), dim=1),
            self.query_embedding,
        )
        conditioning = self.conditioning_norm(
            self.time_embedding(time) + self.action_embedding(action)
        )
        for block in self.flow_blocks:
            query = block(query, context, conditioning)
        patches = self.output(self.output_norm(query))

        patch = self.config.patch_size
        height = self.grid_height // patch
        width = self.grid_width // patch
        patches = patches.reshape(-1, height, width, 2, patch, patch)
        return (
            patches.permute(0, 3, 1, 4, 2, 5)
            .reshape(-1, 2, self.grid_height, self.grid_width)
        )

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        current: Tensor,
        action: Tensor,
    ) -> Tensor:
        return self.velocity(
            state, time, action, self.encode(current)
        )
