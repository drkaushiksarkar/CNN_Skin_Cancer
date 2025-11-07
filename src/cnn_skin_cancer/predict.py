from __future__ import annotations
import typer, yaml, tensorflow as tf, numpy as np
from pathlib import Path
from rich import print

app = typer.Typer(help="Predict class for one or more images")

def _load_image(path, h, w):
    img = tf.keras.utils.load_img(path, target_size=(h, w))
    arr = tf.keras.utils.img_to_array(img) / 255.0
    return arr

@app.command()
def run(model_path: str, images: list[str], config: str = "config/default.yaml"):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    model = tf.keras.models.load_model(model_path)
    xs = np.stack([_load_image(p, cfg["img_height"], cfg["img_width"]) for p in images])
    probs = model.predict(xs, verbose=0)
    for p, pr in zip(images, probs):
        cls = cfg["classes"][int(np.argmax(pr))]
        print(f"[bold]{Path(p).name}[/]: {cls} (p={float(np.max(pr)):.3f})")
