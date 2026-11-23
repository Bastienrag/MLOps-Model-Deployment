"""Réglages communs, sans dépendance lourde : l'image de l'API ne contient pas MLflow."""

import os

MODEL_NAME = os.getenv("MODEL_NAME", "imu-classifier")


def tracking_uri():
    """MLFLOW_TRACKING_URI, sinon localhost sur MLFLOW_PORT (5000)."""
    return os.getenv("MLFLOW_TRACKING_URI") or f"http://localhost:{os.getenv('MLFLOW_PORT', '5000')}"
