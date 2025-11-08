"""Input pipelines and augmentations."""
from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import layers


def augment_layer(cfg: dict) -> tf.keras.Sequential:
    ops: list[tf.keras.layers.Layer] = []
    if cfg.get("flip_left_right", True):
        ops.append(layers.RandomFlip("horizontal"))
    if cfg.get("rotation", 0):
        ops.append(layers.RandomRotation(cfg["rotation"]))
    if cfg.get("zoom", 0):
        ops.append(layers.RandomZoom(cfg["zoom"]))
    if cfg.get("contrast", 0):
        ops.append(layers.RandomContrast(cfg["contrast"]))
    return tf.keras.Sequential(ops, name="augment")


def _dataset_from_dir(directory, image_size, batch_size, seed, shuffle) -> tf.data.Dataset:
    return tf.keras.preprocessing.image_dataset_from_directory(
        directory,
        image_size=image_size,
        seed=seed,
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=shuffle,
    )


def make_datasets(train_dir, val_dir, img_height, img_width, batch_size, seed=123):
    """Create performant tf.data pipelines with caching and prefetch."""

    image_size = (img_height, img_width)
    train = _dataset_from_dir(train_dir, image_size, batch_size, seed, shuffle=True)
    val = _dataset_from_dir(val_dir, image_size, batch_size, seed, shuffle=False)

    options = tf.data.Options()
    options.experimental_deterministic = True
    AUTOTUNE = tf.data.AUTOTUNE

    train = train.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE)
    val = val.cache().prefetch(AUTOTUNE)
    return train.with_options(options), val.with_options(options)
