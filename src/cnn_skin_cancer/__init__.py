"""CNN Skin Cancer package."""

from .config import TrainingConfig, load_config

__all__ = ["TrainingConfig", "load_config", "data", "model", "train", "eval", "predict"]
__version__ = "0.2.0"
