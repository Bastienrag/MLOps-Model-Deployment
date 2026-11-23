"""Met le modèle d'un run dans le registry et lui donne un alias.

    python -m imu.register RUN_ID [alias]       # alias : staging (défaut), puis production

Un run n'a qu'une version dans le registry : promouvoir le même run vers `production` déplace
l'alias sur la version existante au lieu d'en créer une nouvelle.
"""

import sys

import mlflow
from mlflow import MlflowClient

from imu.tracking import MODEL_NAME, connect


def register(run_id, alias):
    client = MlflowClient()
    existing = client.search_model_versions(f"name = '{MODEL_NAME}' and run_id = '{run_id}'")
    if existing:
        version = existing[0].version
    else:
        version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME).version
    client.set_registered_model_alias(MODEL_NAME, alias, version)
    return version


if __name__ == "__main__":
    connect()
    run_id, alias = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else "staging")
    print(f"{MODEL_NAME} v{register(run_id, alias)} est maintenant @{alias}")
