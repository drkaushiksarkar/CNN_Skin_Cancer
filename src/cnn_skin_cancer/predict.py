"""CLI inference helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
import tensorflow as tf
import typer
from rich.console import Console
from rich.table import Table

from .config import load_config

console = Console()
app = typer.Typer(help="Predict skin-lesion classes for supplied images.")


def _load_image(path: str | Path, h: int, w: int) -> np.ndarray:
    img = tf.keras.utils.load_img(path, target_size=(h, w))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return arr


@app.command()
def run(
    model_path: Annotated[str, typer.Argument(help="Trained Keras model")],
    images: Annotated[list[str], typer.Argument(help="Image paths for inference")],
    config: Annotated[
        str, typer.Option("--config", "-c", help="Path to YAML config")
    ] = "config/default.yaml",
    top_k: Annotated[
        int, typer.Option(min=1, max=5, help="Report top-k predictions")
    ] = 3,
    as_json: Annotated[
        bool, typer.Option(help="Emit JSON instead of table output")
    ] = False,
    save_path: Annotated[
        str | None, typer.Option(help="Optional JSON file output")
    ] = None,
):
    cfg = load_config(config)
    model = tf.keras.models.load_model(model_path)
    xs = np.stack([_load_image(p, cfg.img_height, cfg.img_width) for p in images])
    probs = model.predict(xs, verbose=0)

    payload = []
    for path, pr in zip(images, probs):
        order = np.argsort(pr)[::-1][:top_k]
        payload.append(
            {
                "image": Path(path).name,
                "top": [
                    {"class": cfg.classes[int(i)], "prob": float(pr[i])} for i in order
                ],
            }
        )

    if save_path:
        Path(save_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if as_json:
        typer.echo(json.dumps(payload, indent=2))
        return

    table = Table(title="Predictions", caption=f"Model: {Path(model_path).name}")
    table.add_column("Image")
    table.add_column("Prediction")
    table.add_column("Confidence")
    for item in payload:
        top = item["top"][0]
        table.add_row(item["image"], top["class"], f"{top['prob']:.3f}")
    console.print(table)
