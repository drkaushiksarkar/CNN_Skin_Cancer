import subprocess  # nosec B404
import sys
import textwrap


def test_cli_runs_with_default_config(tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        textwrap.dedent(
            """    seed: 123
    img_height: 64
    img_width: 64
    batch_size: 2
    epochs: 1
    optimizer: {name: RMSprop, lr: 0.0001}
    classes: [a,b,c,d,e,f,g,h,i]
    paths: {train_dir: sample/train, val_dir: sample/val, out_dir: runs}
    augment: {flip_left_right: false}
    class_weight: false
    dropout: 0.1
    """
        )
    )
    # only check that package import works (no heavy training)
    out = subprocess.run(  # nosec B603 - controlled command for smoke test
        [sys.executable, "-c", "import cnn_skin_cancer, sys; print('ok')"],
        capture_output=True,
    )
    assert out.returncode == 0  # nosec B101 - pytest assertion preferred
