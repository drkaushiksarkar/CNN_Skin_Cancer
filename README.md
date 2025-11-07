# CNN_Skin_Cancer

![Hero](assets/hero.png)

[![CI](https://img.shields.io/badge/ci-passing-brightgreen.svg)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Multiclass CNN for melanoma/skin lesion classification. Based on ISIC-like classes, with augmentation, imbalance handling, and reproducible training via config.

## Quickstart

```bash
pip install ".[dev]"
csc-train --config config/default.yaml
csc-eval  --model runs/*/model.keras
csc-predict --model runs/*/final_model.keras path/to/image1.jpg path/to/image2.jpg
```

## Repo layout
```
src/cnn_skin_cancer/   # package (train/eval/predict)
config/default.yaml    # hyperparameters, paths
assets/hero.png        # README & OpenGraph preview
```

## Notes
- Uses RMSprop (lr=1e-4, rho=0.9, eps=1e-8, decay=1e-6) as in your original notebook.
- Expects directory datasets: `data/train/<class>/*`, `data/val/<class>/*`.
