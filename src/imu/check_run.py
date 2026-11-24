"""Vérifie qu'un run MLflow passe les seuils d'évaluation. Code de sortie 1 en dessous.

    python -m imu.check_run RUN_ID

Relit dans MLflow les métriques de test, par trajet, journalisées par `imu.track`, et les compare aux seuils
`gate` de params.yaml : la même exigence que le gate de la CI, appliquée avant d'enregistrer le modèle.
"""

import sys
from pathlib import Path

import mlflow
import yaml

from imu.tracking import connect


def check(run_id, gate):
    metrics = mlflow.get_run(run_id).data.metrics
    bacc, f1 = metrics["test_ride_balanced_accuracy"], metrics["test_ride_f1_macro"]
    print(f"run {run_id} : balanced accuracy {bacc:.3f} (seuil {gate['min_balanced_accuracy']}), "
          f"F1 macro {f1:.3f} (seuil {gate['min_f1_macro']})")
    return bacc >= gate["min_balanced_accuracy"] and f1 >= gate["min_f1_macro"]


if __name__ == "__main__":
    connect()
    gate = yaml.safe_load(Path("params.yaml").read_text())["gate"]
    if not check(sys.argv[1], gate):
        sys.exit("Modèle sous les seuils : il n'est pas enregistré")
