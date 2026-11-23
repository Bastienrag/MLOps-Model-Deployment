"""Met le modèle d'un run dans le registry et lui donne un alias.

    python -m imu.register RUN_ID [alias]       # alias : staging (défaut), puis production
"""

import sys

import mlflow
from mlflow import MlflowClient

from imu.tracking import MODEL_NAME, connect

if __name__ == "__main__":
    connect()
    run_id, alias = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "staging")
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    MlflowClient().set_registered_model_alias(MODEL_NAME, alias, version.version)
    print(f"{MODEL_NAME} v{version.version} est maintenant @{alias}")
