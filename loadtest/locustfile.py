"""Test de charge de l'API : de vrais trajets (les 5 exemples, un par classe), un utilisateur = un client.

    uv run --group load locust -f loadtest/locustfile.py --headless -u 10 -r 2 -t 60s \
        --host http://localhost:8080 --only-summary

Un utilisateur envoie un trajet, attend la réponse, puis patiente 0,5 à 1,5 s : N utilisateurs font à peu
près N requêtes par seconde tant que l'API suit, moins dès qu'elle sature.
"""

import json
import random
from pathlib import Path

from locust import HttpUser, between, task

EXAMPLES = Path("src/imu/web/static/examples")


def load_rides():
    rides = []
    for entry in json.loads((EXAMPLES / "index.json").read_text()):
        ex = json.loads((EXAMPLES / entry["file"]).read_text())
        samples = [dict(zip(ex["columns"], row, strict=True)) for row in ex["rows"]]
        rides.append({"ride_id": entry["true_class"], "samples": samples})
    return rides


RIDES = load_rides()


class Rider(HttpUser):
    wait_time = between(0.5, 1.5)

    @task
    def predict(self):
        self.client.post("/v1/predict", json=random.choice(RIDES))
