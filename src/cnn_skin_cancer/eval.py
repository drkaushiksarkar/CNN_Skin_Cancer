from __future__ import annotations
import os, yaml, typer
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import tensorflow as tf

app = typer.Typer(help="Evaluate trained model")

@app.command()
def run(model_path: str, config: str = "config/default.yaml", out_dir: str = "runs/eval"):
    os.makedirs(out_dir, exist_ok=True)
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    model = tf.keras.models.load_model(model_path)
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        cfg["paths"]["val_dir"], image_size=(cfg["img_height"], cfg["img_width"]),
        batch_size=cfg["batch_size"], label_mode="categorical", shuffle=False
    )

    y_true = np.concatenate([y.numpy() for _, y in val_ds], axis=0).argmax(1)
    y_prob = model.predict(val_ds)
    y_pred = y_prob.argmax(1)

    report = classification_report(y_true, y_pred, target_names=cfg["classes"], output_dict=True)
    with open(os.path.join(out_dir, "report.yaml"), "w") as f:
        yaml.safe_dump(report, f)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=cfg["classes"])
    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    out_png = os.path.join("assets", "confusion_matrix.png")
    os.makedirs("assets", exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"Saved confusion matrix to {out_png}")
