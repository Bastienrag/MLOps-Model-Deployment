"""Charge le modèle par alias dans le registry et le score sur le jeu de test.

    python -m imu.score [alias]      # alias : staging (défaut) ou production

Le modèle est chargé par nom et alias, jamais par chemin de fichier. Les statistiques de
normalisation viennent du même run que le modèle (artefact meta.json), pas du dossier local.
"""

import json
import sys
from pathlib import Path

import mlflow
import numpy as np
import yaml
from mlflow import MlflowClient

from imu.baseline import load_split
from imu.evaluate import aggregate_by_ride, metrics
from imu.tracking import MODEL_NAME, connect

if __name__ == "__main__":
    connect()
    alias = sys.argv[1] if len(sys.argv) > 1 else "staging"
    params = yaml.safe_load(Path("params.yaml").read_text())["data"]
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, alias)
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{alias}")
    classes = json.loads(Path(mlflow.artifacts.download_artifacts(
        run_id=version.run_id, artifact_path="meta.json")).read_text())["classes"]

    X, y, ride_id = load_split(params["processed_dir"], "test")
    probs = np.asarray(model.predict(X))
    ride_probs, ride_y = aggregate_by_ride(probs, y, ride_id)
    m = metrics(ride_y, ride_probs.argmax(axis=1), classes)
    print(f"{MODEL_NAME}@{alias} (v{version.version}) sur {len(ride_y)} trajets de test : "
          f"F1 macro {m['f1_macro']:.3f} | balanced acc {m['balanced_accuracy']:.3f}")
