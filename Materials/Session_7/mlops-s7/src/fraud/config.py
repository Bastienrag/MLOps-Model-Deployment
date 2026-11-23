"""Central configuration: params.yaml + environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
PARAMS_PATH = ROOT / "params.yaml"

MODEL_NAME = os.getenv("MODEL_NAME", "fraud-classifier")

KAGGLE_DATASET = "mlg-ulb/creditcardfraud"

PCA_FEATURES = [f"v{i}" for i in range(1, 29)]
FEATURES = PCA_FEATURES + ["amount"]
TARGET = "is_fraud"


@lru_cache
def load_params(path: Path = PARAMS_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)
