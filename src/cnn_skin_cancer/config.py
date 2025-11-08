"""Configuration utilities for the CNN skin cancer application."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field, PositiveInt, validator


class OptimizerConfig(BaseModel):
    """Optimizer hyper-parameters with sensible defaults."""

    name: str = Field(default="RMSprop", description="Keras optimizer name")
    lr: float = Field(default=1e-4, gt=0, description="Learning rate")
    rho: float = Field(default=0.9, gt=0, description="RMSprop rho")
    epsilon: float = Field(default=1e-8, gt=0, description="Numerical stability term")
    decay: float = Field(default=1e-6, ge=0, description="Learning-rate decay")


class AugmentConfig(BaseModel):
    flip_left_right: bool = True
    rotation: float = Field(default=0.0, ge=0.0, le=0.5)
    zoom: float = Field(default=0.0, ge=0.0, le=0.5)
    contrast: float = Field(default=0.0, ge=0.0, le=0.5)


class PathConfig(BaseModel):
    train_dir: Path = Path("data/train")
    val_dir: Path = Path("data/val")
    out_dir: Path = Path("runs")

    @validator("train_dir", "val_dir", pre=True)
    def _as_path(cls, value: str | Path) -> Path:  # noqa: D401
        """Coerce incoming values into Path objects."""

        return Path(value)


class TrainingConfig(BaseModel):
    seed: PositiveInt = 123
    img_height: PositiveInt = 180
    img_width: PositiveInt = 180
    batch_size: PositiveInt = 32
    epochs: PositiveInt = 30
    dropout: float = Field(default=0.3, ge=0.0, le=0.9)
    backbone: str = Field(
        default="efficientnetb0", description="tf.keras.applications backbone"
    )
    fine_tune_at: int | None = Field(
        default=None, ge=0, description="Index of layer to start fine-tuning"
    )
    class_weight: bool = False
    mixed_precision: bool = False
    label_smoothing: float = Field(default=0.0, ge=0.0, le=0.3)
    optimizer: OptimizerConfig = OptimizerConfig()
    augment: AugmentConfig = AugmentConfig()
    paths: PathConfig = PathConfig()
    classes: list[str] = Field(..., min_items=2)
    monitor: str = Field(
        default="val_f1", description="Metric to monitor for callbacks"
    )

    @property
    def num_classes(self) -> int:
        return len(self.classes)


def load_config(path: str | Path) -> TrainingConfig:
    """Load a YAML config file and validate via Pydantic."""

    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return TrainingConfig(**data)
