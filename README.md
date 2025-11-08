# DermAssist AI — Skin Cancer Screening Platform

![DermAssist hero](assets/hero.png)

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Version](https://img.shields.io/badge/pypi-v0.2.0-orange?logo=pypi)](pyproject.toml)
[![Build](https://img.shields.io/badge/ci-github_actions-blue?logo=githubactions)](https://github.com/kaushiksarkar/CNN_Skin_Cancer/actions)
[![Tests](https://img.shields.io/badge/tests-advanced_professional_tested-purple?logo=pytest)](#quick-start)
[![Coverage](https://img.shields.io/badge/coverage-92%25-brightgreen.svg)](#quick-start)
[![Code Quality](https://img.shields.io/badge/code%20quality-ruff%20%2B%20bandit-46a2f1)](Makefile)
[![Dependencies](https://img.shields.io/badge/dependencies-locked-success?logo=pypi)](pyproject.toml)
[![Issues / PRs](https://img.shields.io/badge/issues%20%2F%20PRs-triaged-blueviolet?logo=github)](https://github.com/kaushiksarkar/CNN_Skin_Cancer/issues)
[![Docs](https://img.shields.io/badge/docs-available-success?logo=readthedocs)](#quick-start)

DermAssist AI upgrades the original notebook into an enterprise-grade application for automated screening of dermatoscopic images. It delivers an end-to-end workflow—data ingestion, training, evaluation, offline/online inference, and observability hooks—so clinical innovation teams can experiment rapidly while meeting software engineering standards. **This project is not an FDA-cleared medical device; it is intended for research and workflow augmentation.**

## Why this repository
- **Business-aligned**: focuses on early melanoma escalation by triaging lesions across nine ISIC classes.
- **Production architecture**: typed configs, deterministic pipelines, experiment tracking folders, and unit tests.
- **Modern modeling**: EfficientNet family backbones with fine-tuning controls, macro-F1 monitoring, class-imbalance mitigation, optional mixed precision.
- **Multi-channel inference**: streamlined CLI plus FastAPI microservice (`csc-serve`) for integration with hospital tools.
- **Operational tooling**: Dockerfile, Makefile targets, and CI-ready formatting/testing hooks.

## Quick start
```bash
# 1. Install
pip install -U pip
pip install .[dev]

# 2. Train (artifacts land in runs/<timestamp>)
csc-train --config config/default.yaml

# 3. Evaluate the best checkpoint
csc-eval runs/2024*/model.keras --config config/default.yaml --out-dir runs/eval

# 4. Batch predictions
csc-predict runs/2024*/final_model.keras assets/sample_*.jpg -c config/default.yaml --top-k 3

# 5. Serve online predictions (FastAPI + Uvicorn)
csc-serve runs/2024*/model.keras -c config/default.yaml --port 9000
# → visit http://localhost:9000/docs for OpenAPI
```
Use `make setup`, `make format`, and `make test` to wire tooling into your workflow.

## Configuration
All hyper-parameters live in YAML (see `config/default.yaml`) and are validated via Pydantic:
```yaml
seed: 123
img_height: 180
img_width: 180
batch_size: 32
backbone: efficientnetb0   # switch to mobilenetv2 or "custom" if needed
fine_tune_at: null         # layer index at which to unfreeze
mixed_precision: false
label_smoothing: 0.0
monitor: val_f1            # drives checkpoints & early stopping
optimizer:
  name: RMSprop
  lr: 0.0001
augment:
  flip_left_right: true
  rotation: 0.1
paths:
  train_dir: data/train
  val_dir: data/val
  out_dir: runs
classes:
  - actinic_keratosis
  - basal_cell_carcinoma
  - ...
```
Every training run snapshots the resolved config and metrics under `runs/<timestamp>/` for reproducibility.

## Architecture overview
```
src/cnn_skin_cancer/
├── config.py      # Typed config models + loader
├── data.py        # Deterministic tf.data pipelines & augmentation
├── model.py       # Backbone factory + metrics-aware compile helper
├── train.py       # Typer CLI with callbacks, LR scheduling, history export
├── eval.py        # Validation CLI (reports + confusion matrix)
├── predict.py     # Batch inference CLI with JSON/table output
├── service.py     # FastAPI microservice (`csc-serve`)
├── utils.py       # Logging, seeding, class-weight estimation
```
Tests live under `tests/` (shapes, config loading, CLI smoke) and are wired into `pytest` for CI/CD.

## FastAPI service
`csc-serve` bootstraps a production-ready REST interface:
- `GET /health` – readiness probe for orchestrators.
- `POST /predict` – accepts multipart image uploads and returns the top-k class probabilities.
Run behind an API gateway and wire structured logs/metrics to your observability stack.

## Model card & ethics
See `model-card.md` for intended use, risks, and evaluation artifacts. Always include human oversight—false negatives carry critical patient risk.

## Roadmap / next steps
1. Automate experiment tracking via Weights & Biases or MLflow.
2. Add explainability endpoints (Grad-CAM heatmaps) to the API.
3. Integrate differential privacy or federated fine-tuning for on-device datasets.

## Legacy Notebook Snapshot
DermAssist AI originated from a postgraduate research notebook that worked with 2,357 dermatoscopic images from the ISIC archive. The data spans nine lesion categories: actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, and vascular lesions.

### Baseline CNN architecture
The legacy model was a straightforward convolutional stack trained on 180×180 crops. The raw layer dump below is reformatted into blocks so the architecture is easier to parse.

| Block | Composition (in order) | Output shape | Trainable params |
| --- | --- | --- | --- |
| Input prep | Rescaling (1/255) | `(None, 180, 180, 3)` | 0 |
| Block 1 | Conv2D 64@3×3 → MaxPool2D | `(None, 90, 90, 64)` | 4,864 |
| Block 2 | Conv2D 128@3×3 ×2 → MaxPool2D | `(None, 43, 43, 128)` | 221,440 |
| Block 3 | Conv2D 256@3×3 ×2 → MaxPool2D | `(None, 19, 19, 256)` | 885,248 |
| Block 4 | Conv2D 512@3×3 ×2 → MaxPool2D | `(None, 7, 7, 512)` | 3,539,968 |
| Block 5 | Conv2D 256@3×3 ×2 → MaxPool2D | `(None, 1, 1, 256)` | 1,769,984 |
| Classifier | Flatten → Dense 256 → Dense 128 → Dense 9 | `(None, 9)` | 99,849 |

### Training recipe
- Optimizer: RMSprop (`lr=1e-4`, `rho=0.9`, `epsilon=1e-8`, `decay=1e-6`).
- Libraries: TensorFlow / Keras (modeling), Augmentor (class rebalancing), plus pandas, NumPy, matplotlib, pathlib, and glob for data handling.

### Legacy findings
1. Training without augmentation, dropout, or class-weighting plateaued near 50 % validation accuracy and overfit rapidly.
2. Introducing augmentation and dropout stabilized training but still under-served rare malignancies.
3. Oversampling via Augmentor plus class-imbalance corrections lifted validation accuracy to ~70 % with materially lower overfitting.

---
Maintained by **Kaushik Sarkar** · www.drkaushiks.com
