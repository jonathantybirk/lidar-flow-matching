import tomllib
from dataclasses import dataclass
from pathlib import Path


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
    sensor: SensorConfig
    dataset: DatasetConfig


@dataclass(frozen=True)
class ModelConfig:
    patch_size: int
    width: int
    heads: int
    condition_layers: int
    flow_layers: int


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int
    steps: int
    learning_rate: float
    minimum_learning_rate: float
    warmup_steps: int
    decay_steps: int
    weight_decay: float
    gradient_clip: float
    ema_decay: float
    workers: int
    log_every: int
    validate_every: int
    checkpoint_every: int
    validation_batches: int
    sampling_steps: int
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
