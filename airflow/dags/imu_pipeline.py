"""DAG Airflow : prétraitement -> entraînement -> évaluation -> enregistrement du modèle.

Chaque étape appelle le code du dépôt (`python -m imu.*`) dans l'environnement du projet : le DAG
n'orchestre que l'ordre et les reprises, il ne contient pas de logique d'apprentissage.

    prepare   python -m imu.data       données brutes -> data/processed
    train     python -m imu.track      entraîne le CNN, journalise le run dans MLflow, renvoie son run_id
    evaluate  python -m imu.check_run  échoue si elles passent sous les seuils `gate` de params.yaml
    register  python -m imu.register   modèle du run -> registry, alias `staging`

Paramètres du dépôt (PROJECT_DIR) et serveur MLflow (MLFLOW_PORT ou MLFLOW_TRACKING_URI) : variables
d'environnement du processus Airflow. Voir le README, section « Orchestration (Airflow) ».
"""

import os
import subprocess
from datetime import timedelta
from pathlib import Path

from airflow.sdk import DAG, task

PROJECT_DIR = Path(os.getenv("PROJECT_DIR", Path(__file__).resolve().parents[2]))


def run_in_project(*args):
    """Lance une commande dans le dépôt, affiche sa sortie dans le log de la tâche et la renvoie."""
    proc = subprocess.run(
        ["uv", "run", *args], cwd=PROJECT_DIR, capture_output=True, text=True, check=False
    )
    print(proc.stdout)
    print(proc.stderr)
    proc.check_returncode()
    return proc.stdout


with DAG(
    dag_id="imu_pipeline",
    description="Prétraitement, entraînement, évaluation et enregistrement du modèle IMU",
    schedule=None,  # lancé à la demande (airflow dags trigger) ; rejouable à volonté
    catchup=False,
    max_active_runs=1,  # deux entraînements en parallèle se disputeraient les 2 CPU
    default_args={"retries": 1, "retry_delay": timedelta(minutes=1)},
    tags=["imu", "mlops"],
) as dag:

    @task(execution_timeout=timedelta(minutes=10))
    def prepare():
        """Rejoue le prétraitement : mêmes données brutes et même graine, mêmes fichiers produits."""
        run_in_project("python", "-m", "imu.data")

    @task(execution_timeout=timedelta(minutes=45))
    def train():
        """Entraîne et journalise dans MLflow ; le run_id est transmis aux étapes suivantes."""
        out = run_in_project("python", "-m", "imu.track")
        return next(line.split("=")[1].strip() for line in out.splitlines() if line.startswith("run_id ="))

    @task(execution_timeout=timedelta(minutes=5))
    def evaluate(mlflow_run):
        """Refuse un modèle sous les seuils du gate (métriques de test du run, relues dans MLflow)."""
        run_in_project("python", "-m", "imu.check_run", mlflow_run)
        return mlflow_run

    @task(execution_timeout=timedelta(minutes=5))
    def register(mlflow_run):
        run_in_project("python", "-m", "imu.register", mlflow_run, "staging")

    prepared = prepare()
    trained = train()
    prepared >> trained  # l'entraînement lit data/processed : il attend le prétraitement
    register(evaluate(trained))
