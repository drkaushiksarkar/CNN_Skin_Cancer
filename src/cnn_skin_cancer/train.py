from __future__ import annotations

from pathlib import Path
from typing import Annotated

import tensorflow as tf
import typer
import yaml
from rich.console import Console

from .config import TrainingConfig, load_config
from .data import augment_layer, make_datasets
from .model import build_model, compile_model
from .utils import (
    compute_class_weights,
    configure_logging,
    create_run_dir,
    enable_mixed_precision,
    log_model_summary,
    serialize_history,
    set_seed,
)

console = Console()
app = typer.Typer(help="Train the CNN skin-cancer classifier with enterprise defaults.")


def _persist_config(cfg: TrainingConfig, run_dir: Path) -> None:
    (run_dir / "config.resolved.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json")), encoding="utf-8"
    )


@app.command()
def run(
    config: Annotated[
        str,
        typer.Option(
            "--config",
            "-c",
            help="Path to YAML config file.",
        ),
    ] = "config/default.yaml",
    log_level: Annotated[
        str, typer.Option(help="Python logging level for the run.")
    ] = "INFO",
    dry_run: Annotated[
        bool, typer.Option(help="Build the model and exit (no training).")
    ] = False,
):
    cfg = load_config(config)
    configure_logging(log_level)
    set_seed(cfg.seed)
    enable_mixed_precision(cfg.mixed_precision)

    run_dir = create_run_dir(cfg.paths.out_dir)
    _persist_config(cfg, run_dir)
    console.log(f"Run directory: {run_dir}")

    train_ds, val_ds = make_datasets(
        cfg.paths.train_dir,
        cfg.paths.val_dir,
        cfg.img_height,
        cfg.img_width,
        cfg.batch_size,
        cfg.seed,
    )

    base_model = build_model(
        num_classes=cfg.num_classes,
        img_height=cfg.img_height,
        img_width=cfg.img_width,
        dropout=cfg.dropout,
        backbone=cfg.backbone,
        fine_tune_at=cfg.fine_tune_at,
    )
    aug = augment_layer(cfg.augment.model_dump())
    model = tf.keras.Sequential([aug, base_model], name="augmented_model")
    model.build((None, cfg.img_height, cfg.img_width, 3))
    model = compile_model(
        model,
        cfg.optimizer.model_dump(),
        num_classes=cfg.num_classes,
        label_smoothing=cfg.label_smoothing,
    )
    console.log("Model summary:")
    console.log(log_model_summary(model))

    if dry_run:
        console.log("Dry-run requested; exiting before training.")
        return

    callbacks: list[tf.keras.callbacks.Callback] = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(run_dir / "model.keras"),
            save_best_only=True,
            monitor=cfg.monitor,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True, monitor=cfg.monitor
        ),
        tf.keras.callbacks.TensorBoard(log_dir=str(run_dir / "tb")),
        tf.keras.callbacks.CSVLogger(str(run_dir / "metrics.csv")),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor=cfg.monitor, factor=0.2, patience=4, min_lr=1e-7
        ),
    ]

    class_weight = None
    if cfg.class_weight:
        console.log("Estimating class weights for imbalance mitigation…")
        class_weight = compute_class_weights(train_ds)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.epochs,
        class_weight=class_weight,
        verbose=2,
        callbacks=callbacks,
    )
    model.save(str(run_dir / "final_model.keras"))
    serialize_history(history, run_dir / "history.yaml")
    console.log(f"[green]Training complete[/] · artifacts stored in {run_dir}")
