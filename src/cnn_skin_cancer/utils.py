"""Utility helpers for deterministic training runs and logging."""

from __future__ import annotations

import logging
import os
import random
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import tensorflow as tf


def set_seed(seed: int) -> None:
    """Seed python, numpy, and TensorFlow for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def configure_logging(level: str = "INFO") -> None:
    """Configure process-wide structured logging."""

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def create_run_dir(base_dir: str | Path) -> Path:
    """Create timestamped directory to store artifacts."""

    ts = time.strftime("%Y%m%d-%H%M%S")
    run_dir = Path(base_dir) / ts
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def serialize_history(
    history: tf.keras.callbacks.History, out_path: str | Path
) -> None:
    """Persist Keras training history as float lists."""

    import yaml

    payload = {k: [float(v) for v in history.history[k]] for k in history.history}
    with Path(out_path).open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f)


def enable_mixed_precision(enable: bool) -> None:
    """Turn on TensorFlow mixed precision when requested."""

    if enable:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")


def log_model_summary(model: tf.keras.Model) -> str:
    """Return string summary for audit logging."""

    lines: list[str] = []
    model.summary(print_fn=lines.append)
    return "\n".join(lines)


def compute_class_weights(dataset: tf.data.Dataset) -> dict[int, float]:
    """Estimate class weights with a single pass over the dataset."""

    counts: dict[int, int] = {}
    for _, labels in dataset.unbatch():
        idx = int(tf.argmax(labels).numpy())
        counts[idx] = counts.get(idx, 0) + 1
    total = sum(counts.values())
    return {i: total / (len(counts) * cnt) for i, cnt in counts.items()}


def freeze_backbone(model: tf.keras.Model, layers_to_freeze: Iterable[str]) -> None:
    """Freeze matching layers in-place so we can control fine-tuning."""

    targets = set(layers_to_freeze)
    for layer in model.layers:
        if layer.name in targets:
            layer.trainable = False
