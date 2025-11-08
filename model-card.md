# Model Card — DermAssist AI

- **Version**: 0.2.0
- **Owner**: Kaushik Sarkar (www.drkaushiks.com)
- **Last updated**: 2025-11-08

## Intended Use
Research-grade decision-support aid for triaging dermatoscopic images into nine ISIC-style lesion categories (actinic keratosis, basal cell carcinoma, dermatofibroma, melanoma, nevus, pigmented benign keratosis, seborrheic keratosis, squamous cell carcinoma, vascular lesion). Not cleared as a medical device; requires clinician oversight and regulatory approval before use in patient care.

## Model & Data
- **Architecture**: EfficientNet/B0 backbone with optional fine-tuning, macro-F1 monitoring, label smoothing, class-weight estimation, and mixed-precision toggle.
- **Input**: RGB images resized to 180×180, rescaled to [0,1].
- **Training data**: Expect directory datasets split into `train/` and `val/` folders per class. Source data mirrors ISIC distribution (≈2.3k images) with augmentation (flip, rotation, zoom, contrast).
- **Outputs**: Probability distribution over the nine lesion classes.

## Performance Reporting
Run `csc-eval` to produce:
- `report.yaml`: precision / recall / F1 per class + macro/micro aggregates (Sklearn `classification_report`).
- `assets/confusion_matrix.png`: confusion matrix saved with consistent labels.

Key KPI for model selection: **macro F1** (targets balanced sensitivity across rare lesion types). Track `val_accuracy`, `val_top3_acc`, and `val_auc` as supporting signals.

## Ethical Considerations & Risks
- **Bias**: Source data under-represents darker skin tones and rare lesion morphologies, leading to potential disparate performance.
- **False negatives**: Missing a melanoma case is the highest-risk failure mode; deploy with clinician review workflows.
- **Data privacy**: Images may contain PHI (tattoos, backgrounds). Apply de-identification before training.

## Deployment Guidance
- Batch scoring via `csc-predict` (JSON/table output).
- Real-time scoring via FastAPI (`csc-serve`) with `/health` and `/predict` endpoints; wrap behind authentication, logging, and monitoring.
- Container image provided in `Dockerfile`; set the run-time model/config paths via bind mounts or environment-specific configs.

## Limitations & Future Work
- No federated learning or continual updates; retrain offline as new cohorts arrive.
- No saliency/explainability artifacts; integrate Grad-CAM overlays before clinical pilots.
- Does not perform quality checks on input images (focus, glare). Consider adding QA heuristics.

## Contact
Questions or bug reports: [@drkaushiksarkar](https://www.linkedin.com/in/kaushiksarkar) or raise an issue in this repository.
