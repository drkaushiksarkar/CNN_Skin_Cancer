from __future__ import annotations

from typing import Callable

import tensorflow_addons as tfa
from tensorflow import keras
from tensorflow.keras import layers

_BACKBONES: dict[str, Callable[..., keras.Model]] = {
    "efficientnetb0": keras.applications.EfficientNetB0,
    "efficientnetv2b0": keras.applications.EfficientNetV2B0,
    "mobilenetv2": keras.applications.MobileNetV2,
}


def build_model(
    num_classes: int,
    img_height: int = 180,
    img_width: int = 180,
    dropout: float = 0.3,
    backbone: str = "efficientnetb0",
    fine_tune_at: int | None = None,
) -> keras.Model:
    """Factory for the classifier supporting modern backbones."""

    inputs = keras.Input(shape=(img_height, img_width, 3), name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)

    backbone = backbone.lower()
    base_model = None
    if backbone in _BACKBONES:
        base_model = _BACKBONES[backbone](
            include_top=False,
            weights=None,
            input_shape=(img_height, img_width, 3),
        )
        base_model.trainable = False
        if fine_tune_at is not None:
            for layer in base_model.layers[:fine_tune_at]:
                layer.trainable = False
            for layer in base_model.layers[fine_tune_at:]:
                layer.trainable = True
        x = base_model(x, training=False)
        x = layers.GlobalAveragePooling2D(name="gap")(x)
    else:
        x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation="relu")(x)
        x = layers.Conv2D(128, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(256, 3, activation="relu")(x)
        x = layers.Conv2D(256, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(512, 3, activation="relu")(x)
        x = layers.Conv2D(512, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(256, 3, activation="relu")(x)
        x = layers.Conv2D(256, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)

    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    outputs = layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    return keras.Model(inputs, outputs, name=f"cnn_skin_cancer_{backbone}")


def compile_model(
    model: keras.Model,
    opt_cfg: dict,
    num_classes: int,
    label_smoothing: float = 0.0,
) -> keras.Model:
    """Compile model with advanced metrics."""

    name = opt_cfg.get("name", "RMSprop").lower()
    if name == "rmsprop":
        rms_kwargs = {
            "learning_rate": opt_cfg.get("lr", 1e-4),
            "rho": opt_cfg.get("rho", 0.9),
            "epsilon": opt_cfg.get("epsilon", 1e-8),
        }
        decay = opt_cfg.get("decay")
        if decay:
            rms_kwargs["decay"] = decay
            opt = keras.optimizers.legacy.RMSprop(**rms_kwargs)
        else:
            opt = keras.optimizers.RMSprop(**rms_kwargs)
    else:
        opt = keras.optimizers.Adam(learning_rate=opt_cfg.get("lr", 1e-4))

    loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    metrics = [
        keras.metrics.CategoricalAccuracy(name="accuracy"),
        keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_acc"),
        tfa.metrics.F1Score(name="f1", num_classes=num_classes, average="macro"),
        keras.metrics.AUC(curve="ROC", multi_label=True, name="auc"),
    ]
    model.compile(optimizer=opt, loss=loss, metrics=metrics)
    return model
