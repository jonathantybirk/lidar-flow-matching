import tomllib
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SurfaceConfig:
    candidate_points: int
    target_points: int
    normal_weight: float


@dataclass(frozen=True)
class SensorConfig:
    height: int
    width: int
    horizontal_fov_degrees: float
    vertical_fov_degrees: float
    distance: float
    minimum_elevation_degrees: float
    maximum_elevation_degrees: float


@dataclass(frozen=True)
class DatasetConfig:
    training_views: int
    evaluation_views: int
    validation_fraction: float
    test_fraction: float
    seed: int


@dataclass(frozen=True)
class DataConfig:
    surface: SurfaceConfig
    sensor: SensorConfig
    dataset: DatasetConfig


@dataclass(frozen=True)
class ModelConfig:
    points: int
    condition_points: int
    width: int
    heads: int
    condition_layers: int
    flow_layers: int
    neighbors: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    steps: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_steps: int
    weight_decay: float
    gradient_clip: float
    chamfer_weight: float
    observation_weight: float
    ema_decay: float
    workers: int
    log_every: int
    validate_every: int
    checkpoint_every: int
    validation_batches: int
    early_stopping_patience: int
    early_stopping_min_delta: float
    seed: int


@dataclass(frozen=True)
class ExperimentConfig:
    model: ModelConfig
    training: TrainingConfig


def load_data_config(path: Path) -> DataConfig:
    with path.open("rb") as file:
        values = tomllib.load(file)

    return DataConfig(
        surface=SurfaceConfig(**values["surface"]),
        sensor=SensorConfig(**values["sensor"]),
        dataset=DatasetConfig(**values["dataset"]),
    )


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.open("rb") as file:
        values = tomllib.load(file)

    return ExperimentConfig(
        model=ModelConfig(**values["model"]),
        training=TrainingConfig(**values["training"]),
    )
