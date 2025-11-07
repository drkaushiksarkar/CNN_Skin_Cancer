from __future__ import annotations
import os, time, yaml, typer
from rich import print
import tensorflow as tf
from .model import build_model, compile_model
from .data import make_datasets, augment_layer

app = typer.Typer(help="Train CNN skin-cancer classifier")

@app.command()
def run(config: str = "config/default.yaml"):
    with open(config, "r") as f:
        cfg = yaml.safe_load(f)

    os.makedirs(cfg["paths"]["out_dir"], exist_ok=True)
    run_dir = os.path.join(cfg["paths"]["out_dir"], time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(run_dir, exist_ok=True)

    train_ds, val_ds = make_datasets(
        cfg["paths"]["train_dir"], cfg["paths"]["val_dir"],
        cfg["img_height"], cfg["img_width"], cfg["batch_size"], cfg["seed"]
    )

    model = build_model(num_classes=len(cfg["classes"]),
                        img_height=cfg["img_height"], img_width=cfg["img_width"],
                        dropout=cfg["dropout"])
    model = compile_model(model, cfg["optimizer"])

    aug = augment_layer(cfg["augment"])
    model_aug = tf.keras.Sequential([aug, model], name="augmented_model")

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(os.path.join(run_dir, "model.keras"),
                                           save_best_only=True, monitor="val_accuracy", mode="max"),
        tf.keras.callbacks.EarlyStopping(patience=6, restore_best_weights=True, monitor="val_accuracy"),
        tf.keras.callbacks.TensorBoard(log_dir=os.path.join(run_dir, "tb")),
    ]

    class_weight = None
    if cfg.get("class_weight", False):
        print("[yellow]Estimating class weights (one pass) …[/]")
        counts = {}
        for _, y in train_ds.unbatch():
            idx = int(tf.argmax(y).numpy())
            counts[idx] = counts.get(idx, 0) + 1
        total = sum(counts.values())
        class_weight = {i: total/(len(counts)*cnt) for i, cnt in counts.items()}

    history = model_aug.fit(
        train_ds, validation_data=val_ds, epochs=cfg["epochs"],
        class_weight=class_weight, verbose=2, callbacks=callbacks
    )
    model.save(os.path.join(run_dir, "final_model.keras"))
    with open(os.path.join(run_dir, "history.yaml"), "w") as f:
        yaml.safe_dump({k: [float(v) for v in history.history[k]] for k in history.history}, f)

    print(f"[green]Done. Artifacts in[/] {run_dir}")
