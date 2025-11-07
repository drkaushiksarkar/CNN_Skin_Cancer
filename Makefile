setup: ; pip install -U pip && pip install ".[dev]" && pre-commit install
format: ; ruff --fix . && black .
test: ; pytest -q
train: ; csc-train --config config/default.yaml
