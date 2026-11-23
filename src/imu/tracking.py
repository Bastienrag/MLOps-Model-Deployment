"""Connexion à MLflow et empreintes du code et des données d'un entraînement."""

import os
import subprocess
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("MLFLOW_ENABLE_ARTIFACTS_PROGRESS_BAR", "false")
# Les artefacts passent par le serveur MLflow (--serve-artifacts). Sans cela, le serveur renvoie des URL
# présignées vers s3:8333, une adresse interne à Docker que le poste du développeur ne joint pas.
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_DOWNLOAD", "false")
os.environ.setdefault("MLFLOW_ENABLE_PROXY_MULTIPART_UPLOAD", "false")

import mlflow  # noqa: E402
import yaml  # noqa: E402

MODEL_NAME = os.getenv("MODEL_NAME", "imu-classifier")


def connect():
    """Pointe vers le serveur MLflow : MLFLOW_TRACKING_URI, sinon localhost sur MLFLOW_PORT (5000)."""
    uri = os.getenv("MLFLOW_TRACKING_URI") or f"http://localhost:{os.getenv('MLFLOW_PORT', '5000')}"
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(os.getenv("EXPERIMENT", "imu"))
    return uri


def git(*args):
    return subprocess.check_output(["git", *args], text=True).strip()


def git_dirty():
    """True si des fichiers du projet (hors Materials/) ne sont pas commités. Le .env est ignoré par Git."""
    return bool(git("status", "--porcelain", "--", ".", ":!Materials"))


def dvc_md5(dvc_file):
    """Empreinte d'un fichier suivi par DVC, lue dans le petit fichier .dvc que Git garde."""
    return yaml.safe_load(Path(dvc_file).read_text())["outs"][0]["md5"]


def data_fingerprints(params, lock_file="dvc.lock"):
    """Empreintes des données brutes, des labels et des jeux prétraités (stage `prepare`)."""
    lock = yaml.safe_load(Path(lock_file).read_text())
    outs = lock["stages"]["prepare"]["outs"]
    processed = next(o["md5"] for o in outs if o["path"] == params["processed_dir"])
    return {
        "data_raw_md5": dvc_md5(f"{params['raw_path']}.dvc"),
        "data_labels_md5": dvc_md5(f"{params['labels_path']}.dvc"),
        "data_processed_md5": processed,
    }
