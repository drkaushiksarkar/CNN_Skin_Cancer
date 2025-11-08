from pathlib import Path

from cnn_skin_cancer.config import TrainingConfig, load_config


def test_load_config_roundtrip(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        """
seed: 1
img_height: 64
img_width: 64
batch_size: 4
epochs: 2
dropout: 0.1
backbone: custom
classes: [a, b]
paths:
  train_dir: data/train
  val_dir: data/val
  out_dir: runs
optimizer: {name: RMSprop, lr: 0.0001}
augment: {flip_left_right: false}
        """
    )
    cfg = load_config(cfg_path)
    assert isinstance(cfg, TrainingConfig)
    dumped = cfg.model_dump()
    assert dumped["img_height"] == 64
    assert dumped["classes"] == ["a", "b"]
