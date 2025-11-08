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

<<<<<<< HEAD
## Table of Contents
* [General Info](#general-information)
* [Technologies Used](#technologies-used)
* [Conclusions](#conclusions)
* [Acknowledgements](#acknowledgements)


## General Information
- Melanoma is a type of cancer that can be deadly if not detected early. It accounts for 75% of skin cancer deaths. A solution that can evaluate images and alert dermatologists about the presence of melanoma has the potential to reduce a lot of manual effort needed in diagnosis.
- The dataset consists of 2357 images of malignant and benign oncological diseases, which were formed from the International Skin Imaging Collaboration (ISIC). All images were sorted according to the classification taken with ISIC, and all subsets were divided into the same number of images, with the exception of melanomas and moles, whose images are slightly dominant.
- The data set contains the following diseases:

* Actinic keratosis
* Basal cell carcinoma
* Dermatofibroma
* Melanoma
* Nevus
* Pigmented benign keratosis
* Seborrheic keratosis
* Squamous cell carcinoma
* Vascular lesion
- I have used a CNN model having the following architecture:

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 rescaling (Rescaling)       (None, 180, 180, 3)       0         
                                                                 
 conv2d (Conv2D)             (None, 180, 180, 64)      4864      
                                                                 
 max_pooling2d (MaxPooling2D  (None, 90, 90, 64)       0         
 )                                                               
                                                                 
 conv2d_1 (Conv2D)           (None, 88, 88, 128)       73856     
                                                                 
 conv2d_2 (Conv2D)           (None, 86, 86, 128)       147584    
                                                                 
 max_pooling2d_1 (MaxPooling  (None, 43, 43, 128)      0         
 2D)                                                             
                                                                 
 conv2d_3 (Conv2D)           (None, 41, 41, 256)       295168    
                                                                 
 conv2d_4 (Conv2D)           (None, 39, 39, 256)       590080    
                                                                 
 max_pooling2d_2 (MaxPooling  (None, 19, 19, 256)      0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 17, 17, 512)       1180160   
                                                                 
 conv2d_6 (Conv2D)           (None, 15, 15, 512)       2359808   
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 7, 7, 512)        0         
 2D)                                                             
                                                                 
 conv2d_7 (Conv2D)           (None, 5, 5, 256)         1179904   
                                                                 
 conv2d_8 (Conv2D)           (None, 3, 3, 256)         590080    
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 1, 1, 256)        0         
 2D)                                                             
                                                                 
 flatten (Flatten)           (None, 256)               0         
                                                                 
 dense (Dense)               (None, 256)               65792     
                                                                 
 dense_1 (Dense)             (None, 128)               32896     
                                                                 
 dense_2 (Dense)             (None, 9)                 1161      
                                                                 
=================================================================
- I have used RMSprop optimizer and used learning rate 0.0001, rho=0.9, epsilon=1e-08, decay=1e-6.

## Conclusions
- Conclusion 1 – CNN without augmentation, dropout, and class imbalance correction resulted in 50% validation accuracy with a tendancy to overfit.
- Conclusion 2 – CNN with augmentation, after applying dropout layers reduced the tendency to overfit.
- Conclusion 3 – CNN with class imbalance correction improved accuracy and reduced overfitting. Final accuracy was 70% after addressing class imbalance using Augmentor.


## Technologies Used
- library - pathlib, glob, matplotlib, numpy, pandas, tensorflow, keras, Augmentor

## Acknowledgements
- This is an academic project, inspired by UpGrad
- This project was created to fulfil PGP AIML Requirement.


## Contact
Created by [@drkaushiksarkar] - feel free to contact me! www.drkaushiks.com
>>>>>>> e0eb089f3e7da80f9bbe9367d1f737631cf94879
=======
---
Maintained by **Kaushik Sarkar** · www.drkaushiks.com
>>>>>>> 4bf2664 (updated repo structure)
