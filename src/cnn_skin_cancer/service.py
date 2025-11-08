"""FastAPI inference service for the CNN classifier."""
from __future__ import annotations

from typing import List

import numpy as np
import tensorflow as tf
import typer
import uvicorn
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from .config import TrainingConfig, load_config

cli = typer.Typer(help="Serve the model behind a FastAPI endpoint.")


class Prediction(BaseModel):
    label: str
    probability: float


class PredictionResponse(BaseModel):
    top: List[Prediction]


def _prepare(image_bytes: bytes, cfg: TrainingConfig) -> np.ndarray:
    image = tf.io.decode_image(image_bytes, channels=3)
    image = tf.image.resize(image, (cfg.img_height, cfg.img_width))
    image = tf.cast(image, tf.float32) / 255.0
    return image.numpy()


def create_app(model_path: str, cfg: TrainingConfig, top_k: int) -> FastAPI:
    model = tf.keras.models.load_model(model_path)
    app = FastAPI(title="Skin Cancer Screening API", version="0.2.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictionResponse)
    async def predict(file: UploadFile = File(...)) -> PredictionResponse:
        data = await file.read()
        arr = _prepare(data, cfg)
        probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
        order = np.argsort(probs)[::-1][:top_k]
        return PredictionResponse(
            top=[
                Prediction(label=cfg.classes[int(idx)], probability=float(probs[idx]))
                for idx in order
            ]
        )

    return app


@cli.command()
def serve(
    model_path: str = typer.Argument(..., help="Path to .keras or SavedModel"),
    config: str = typer.Option("config/default.yaml", "--config", "-c"),
    host: str = typer.Option("0.0.0.0", help="Host/IP to bind"),
    port: int = typer.Option(8000, help="TCP port"),
    top_k: int = typer.Option(3, help="How many classes to return"),
):
    cfg = load_config(config)
    app = create_app(model_path, cfg, top_k)
    uvicorn.run(app, host=host, port=port)
