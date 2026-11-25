"""Surveillance de l'API : métriques Prometheus sur 4 couches et un log JSON par événement.

1. infrastructure : requêtes, erreurs et latence par route ;
2. entrées : quelques statistiques du signal reçu (longueur, agitation, vitesse de rotation, trous) ;
3. prédictions : part de chaque classe prédite et confiance ;
4. résultat métier : vérité terrain reçue plus tard par /v1/feedback, croisée avec la classe prédite.

Aucun label n'a de cardinalité non bornée : les routes sont celles du code (`/v1/predict`, jamais l'URL
brute), les classes sont au nombre de 5. Les identifiants de trajet n'apparaissent jamais en clair : le
log ne contient que leur empreinte salée (`LOG_HASH_SALT`).
"""

import hashlib
import json
import os
import sys
import threading
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from prometheus_client import Counter, Gauge, Histogram

# --- couche 1 : infrastructure -------------------------------------------------------------------
REQUESTS = Counter("imu_http_requests_total", "Requêtes HTTP", ["method", "route", "status"])
LATENCY = Histogram(
    "imu_http_request_duration_seconds", "Durée de traitement d'une requête", ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5),
)
MODEL_INFO = Gauge("imu_model_info", "Version du modèle chargé (valeur 1)", ["version"])

# --- couche 2 : distribution des entrées ---------------------------------------------------------
INPUT_SAMPLES = Histogram(
    "imu_input_samples", "Nombre de mesures d'un trajet reçu",
    buckets=(256, 500, 1000, 1500, 2000, 3000, 5000, 10000),
)
INPUT_ACCEL_STD = Histogram(
    "imu_input_accel_std", "Écart-type de la norme de l'accélération (m/s²) : l'agitation du trajet",
    buckets=(0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5),
)
INPUT_GYRO_RMS = Histogram(
    "imu_input_gyro_rms", "Valeur efficace de la norme de la vitesse angulaire (°/s)",
    buckets=(5, 10, 20, 30, 40, 60, 80, 120, 200),
)
INPUT_MISSING = Histogram(
    "imu_input_missing_fraction", "Part de lignes manquantes d'un trajet reçu",
    buckets=(0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.25, 0.5),
)

# --- couche 3 : distribution des prédictions -----------------------------------------------------
PREDICTIONS = Counter("imu_predictions_total", "Prédictions rendues, par classe", ["predicted_class"])
CONFIDENCE = Histogram(
    "imu_prediction_confidence", "Probabilité de la classe prédite",
    buckets=(0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

# --- couche 4 : résultat métier (vérité terrain reçue après coup) --------------------------------
FEEDBACK = Counter(
    "imu_feedback_total", "Vérités terrain reçues, par classe prédite et classe réelle",
    ["predicted_class", "true_class"],
)
FEEDBACK_CORRECT = Counter(
    "imu_feedback_correct_total", "Vérités terrain où la prédiction était juste, par classe réelle",
    ["true_class"],
)


def input_features(values):
    """Statistiques d'un trajet (temps, 6) avec NaN pour les lignes manquantes."""
    accel = np.linalg.norm(values[:, :3], axis=1)
    gyro = np.linalg.norm(values[:, 3:], axis=1)
    return {
        "n_samples": int(len(values)),
        "accel_std": float(np.nanstd(accel)),
        "gyro_rms": float(np.sqrt(np.nanmean(gyro**2))),
        "missing_fraction": float(np.isnan(values[:, 0]).mean()),
    }


def observe_input(features):
    INPUT_SAMPLES.observe(features["n_samples"])
    INPUT_ACCEL_STD.observe(features["accel_std"])
    INPUT_GYRO_RMS.observe(features["gyro_rms"])
    INPUT_MISSING.observe(features["missing_fraction"])


def hash_id(ride_id):
    """Empreinte salée d'un identifiant : relie les événements d'un même trajet sans le révéler."""
    if ride_id is None:
        return None
    salt = os.getenv("LOG_HASH_SALT", "imu")
    return hashlib.sha256(f"{salt}:{ride_id}".encode()).hexdigest()[:16]


class EventLog:
    """Une ligne JSON par événement : dans PREDICTION_LOG si défini, sinon sur la sortie standard."""

    def __init__(self):
        self.lock = threading.Lock()

    def write(self, event, **fields):
        record = {"ts": datetime.now(UTC).isoformat(timespec="milliseconds"), "event": event, **fields}
        line = json.dumps(record, ensure_ascii=False)
        path = os.getenv("PREDICTION_LOG")
        with self.lock:
            if path:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                with open(path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            else:
                print(line, file=sys.stdout, flush=True)


events = EventLog()
