from __future__ import annotations
import tensorflow as tf
from tensorflow.keras import layers

def augment_layer(cfg):
    ops = []
    if cfg.get("flip_left_right", True):
        ops.append(layers.RandomFlip("horizontal"))
    if cfg.get("rotation", 0):
        ops.append(layers.RandomRotation(cfg["rotation"]))
    if cfg.get("zoom", 0):
        ops.append(layers.RandomZoom(cfg["zoom"]))
    if cfg.get("contrast", 0):
        ops.append(layers.RandomContrast(cfg["contrast"]))
    return tf.keras.Sequential(ops, name="augment")

def make_datasets(train_dir, val_dir, img_height, img_width, batch_size, seed=123):
    train = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir, image_size=(img_height, img_width), seed=seed,
        batch_size=batch_size, label_mode="categorical"
    )
    val = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir, image_size=(img_height, img_width), seed=seed,
        batch_size=batch_size, label_mode="categorical"
    )
    AUTOTUNE = tf.data.AUTOTUNE
    return (
        train.cache().shuffle(1000, seed=seed).prefetch(AUTOTUNE),
        val.cache().prefetch(AUTOTUNE),
    )
