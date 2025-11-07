from __future__ import annotations
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_model(num_classes: int, img_height: int = 180, img_width: int = 180, dropout: float = 0.3):
    inputs = keras.Input(shape=(img_height, img_width, 3))
    x = layers.Rescaling(1./255)(inputs)

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
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs, name="cnn_skin_cancer")

def compile_model(model: keras.Model, opt_cfg: dict):
    name = opt_cfg.get("name", "RMSprop").lower()
    if name == "rmsprop":
        opt = keras.optimizers.RMSprop(
            learning_rate=opt_cfg.get("lr", 1e-4),
            rho=opt_cfg.get("rho", 0.9),
            epsilon=opt_cfg.get("epsilon", 1e-8),
            decay=opt_cfg.get("decay", 1e-6),
        )
    else:
        opt = keras.optimizers.Adam(learning_rate=1e-4)
    model.compile(optimizer=opt, loss="categorical_crossentropy", metrics=["accuracy"])
    return model
