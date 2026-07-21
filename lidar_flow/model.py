import math

import torch
from torch import Tensor, nn

from lidar_flow.config import ModelConfig


class LocalPointBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(width + 4, width),
            nn.SiLU(),
            nn.Linear(width, width),
        )
        self.norm = nn.LayerNorm(width)

    def forward(
        self, features: Tensor, xyz: Tensor, mask: Tensor, neighbors: Tensor
    ) -> Tensor:
        batch = torch.arange(len(features), device=features.device)[:, None, None]
        neighbor_features = features[batch, neighbors]
        offsets = xyz[batch, neighbors] - xyz[:, :, None]
        distances = offsets.norm(dim=-1, keepdim=True)
        messages = self.message(
            torch.cat((neighbor_features, offsets, distances), dim=-1)
        )
        update = messages.amax(dim=2)
        return self.norm(features + update) * mask[:, :, None]


class ConditionEncoder(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.neighbors = config.neighbors
        self.point_embedding = nn.Sequential(
            nn.Linear(10, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.local_blocks = nn.ModuleList(
            [LocalPointBlock(config.width) for _ in range(2)]
        )
        self.sensor_embedding = nn.Sequential(
            nn.Linear(9, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.global_blocks = nn.ModuleList(
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
        self.norm = nn.LayerNorm(config.width)

    def forward(
        self, point_features: Tensor, xyz: Tensor, mask: Tensor, sensor: Tensor
    ) -> tuple[Tensor, Tensor]:
        distances = torch.cdist(xyz.float(), xyz.float())
        distances.masked_fill_(~mask[:, None], torch.inf)
        neighbors = distances.topk(self.neighbors, largest=False).indices
        features = self.point_embedding(point_features) * mask[:, :, None]
        for block in self.local_blocks:
            features = block(features, xyz, mask, neighbors)

        sensor_token = self.sensor_embedding(sensor)[:, None]
        features = torch.cat((features, sensor_token), dim=1)
        condition_mask = torch.cat(
            (mask, torch.ones((len(mask), 1), dtype=torch.bool, device=mask.device)),
            dim=1,
        )
        for block in self.global_blocks:
            features = block(features, src_key_padding_mask=~condition_mask)
            features = features * condition_mask[:, :, None]
        return self.norm(features), condition_mask


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
        embedding = torch.cat((angles.cos(), angles.sin()), dim=-1)
        return self.mlp(embedding)


def _modulate(value: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    return value * (1 + scale[:, None]) + shift[:, None]


class FlowBlock(nn.Module):
    def __init__(self, width: int, heads: int) -> None:
        super().__init__()
        self.condition_cross_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.condition_self_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.condition_mlp_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.point_cross_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.point_mlp_norm = nn.LayerNorm(width, elementwise_affine=False)
        self.condition_cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.condition_self = nn.MultiheadAttention(width, heads, batch_first=True)
        self.point_cross = nn.MultiheadAttention(width, heads, batch_first=True)
        self.condition_mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.point_mlp = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.condition_modulation = nn.Linear(width, 9 * width)
        self.point_modulation = nn.Linear(width, 6 * width)
        nn.init.zeros_(self.condition_modulation.weight)
        nn.init.zeros_(self.condition_modulation.bias)
        nn.init.zeros_(self.point_modulation.weight)
        nn.init.zeros_(self.point_modulation.bias)

    def forward(
        self, points: Tensor, condition: Tensor, condition_mask: Tensor, time: Tensor
    ) -> tuple[Tensor, Tensor]:
        condition_values = self.condition_modulation(time).chunk(9, dim=-1)
        point_values = self.point_modulation(time).chunk(6, dim=-1)
        cs1, cc1, cg1, cs2, cc2, cg2, cs3, cc3, cg3 = condition_values
        ps1, pc1, pg1, ps2, pc2, pg2 = point_values

        query = _modulate(self.condition_cross_norm(condition), cs1, cc1)
        update = self.condition_cross(query, points, points, need_weights=False)[0]
        condition = condition + cg1[:, None] * update

        query = _modulate(self.condition_self_norm(condition), cs2, cc2)
        update = self.condition_self(
            query,
            query,
            query,
            key_padding_mask=~condition_mask,
            need_weights=False,
        )[0]
        condition = condition + cg2[:, None] * update
        update = self.condition_mlp(
            _modulate(self.condition_mlp_norm(condition), cs3, cc3)
        )
        condition = (condition + cg3[:, None] * update) * condition_mask[:, :, None]

        query = _modulate(self.point_cross_norm(points), ps1, pc1)
        update = self.point_cross(
            query,
            condition,
            condition,
            key_padding_mask=~condition_mask,
            need_weights=False,
        )[0]
        points = points + pg1[:, None] * update
        update = self.point_mlp(_modulate(self.point_mlp_norm(points), ps2, pc2))
        points = points + pg2[:, None] * update
        return points, condition


class PointFlowTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.condition_encoder = ConditionEncoder(config)
        self.point_embedding = nn.Sequential(
            nn.Linear(6, config.width),
            nn.SiLU(),
            nn.Linear(config.width, config.width),
        )
        self.time_embedding = TimestepEmbedding(config.width)
        self.blocks = nn.ModuleList(
            [FlowBlock(config.width, config.heads) for _ in range(config.flow_layers)]
        )
        self.output_norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, 3)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def encode_condition(
        self,
        observed_features: Tensor,
        observed_xyz: Tensor,
        observed_mask: Tensor,
        sensor_features: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return self.condition_encoder(
            observed_features, observed_xyz, observed_mask, sensor_features
        )

    def velocity(
        self,
        state: Tensor,
        time: Tensor,
        origin: Tensor,
        condition: Tensor,
        condition_mask: Tensor,
    ) -> Tensor:
        points = self.point_embedding(torch.cat((state, origin), dim=-1))
        time_embedding = self.time_embedding(time)
        for block in self.blocks:
            points, condition = block(points, condition, condition_mask, time_embedding)
        return self.output(self.output_norm(points))

    def forward(
        self,
        state: Tensor,
        time: Tensor,
        origin: Tensor,
        observed_features: Tensor,
        observed_xyz: Tensor,
        observed_mask: Tensor,
        sensor_features: Tensor,
    ) -> Tensor:
        condition, condition_mask = self.encode_condition(
            observed_features, observed_xyz, observed_mask, sensor_features
        )
        return self.velocity(state, time, origin, condition, condition_mask)
