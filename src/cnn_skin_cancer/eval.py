"""Model evaluation CLI."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import typer
from rich.console import Console
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import yaml

from .config import load_config

console = Console()
app = typer.Typer(help="Evaluate trained models against the validation split.")


@app.command()
def run(
    model_path: str = typer.Argument(..., help="Path to saved Keras model"),
    config: str = typer.Option("config/default.yaml", "--config", "-c"),
    out_dir: str = typer.Option("runs/eval", help="Where to store reports and figures"),
    save_confusion: str = typer.Option("assets/confusion_matrix.png", help="PNG path"),
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    cfg = load_config(config)

    model = tf.keras.models.load_model(model_path)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        cfg.paths.val_dir,
        image_size=(cfg.img_height, cfg.img_width),
        batch_size=cfg.batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0).argmax(1)
    y_prob = model.predict(val_ds)
    y_pred = y_prob.argmax(1)

    report = classification_report(
        y_true,
        y_pred,
        target_names=cfg.classes,
        output_dict=True,
        zero_division=0,
    )
    (out_path / "report.yaml").write_text(yaml.safe_dump(report), encoding="utf-8")

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=cfg.classes)
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    Path(save_confusion).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_confusion, dpi=150)
    console.log(f"Evaluation artifacts written to {out_path}")
