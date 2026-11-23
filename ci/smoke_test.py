"""Smoke test d'une image en marche : l'API répond et classe un trajet.

    python ci/smoke_test.py http://localhost:8000 <version attendue>

Utilise uniquement la bibliothèque standard. Code de sortie 1 au premier écart.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

CLASSES = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]
EXAMPLE = Path("src/imu/web/static/examples/reckless.json")


def call(url, payload=None, attempts=5):
    """Appelle l'API. Une coupure de connexion est réessayée (juste après un déploiement, la redirection de
    port garde un instant des connexions vers les anciens pods) ; une réponse HTTP, même en erreur, non."""
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data, {"Content-Type": "application/json"})
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=30) as r:
                return r.status, json.load(r), r.headers
        except urllib.error.HTTPError as err:
            return err.code, json.load(err), err.headers
        except OSError:
            if attempt == attempts - 1:
                raise
            time.sleep(2)


def check(ok, message):
    print(("OK   " if ok else "ÉCHEC"), message)
    if not ok:
        sys.exit(1)


def main(base, version):
    for _ in range(45):  # le démarrage charge torch : jusqu'à 90 s
        try:
            status, health, _ = call(f"{base}/health")
            if status == 200:
                break
        except OSError:
            pass
        time.sleep(2)
    else:
        check(False, "l'API ne répond pas sur /health")

    check(health["model_loaded"] and health["status"] == "ok", f"/health : modèle chargé ({health})")
    check(health["model_version"] == version, f"version du modèle = {version}")

    ex = json.loads(EXAMPLE.read_text())
    ride = {"ride_id": "smoke", "samples": [dict(zip(ex["columns"], row, strict=True)) for row in ex["rows"]]}
    status, body, headers = call(f"{base}/v1/predict", ride)
    check(status == 200, f"/v1/predict répond 200 (reçu {status})")
    check(body["predicted_class"] in CLASSES, f"catégorie valide : {body['predicted_class']}")
    check(abs(sum(body["probabilities"].values()) - 1) < 1e-4, "les probabilités somment à 1")
    check(body["model_version"] == version == headers["X-Model-Version"], "version du modèle dans la réponse")

    status, body, _ = call(f"{base}/v1/predict", {"samples": ride["samples"][:10]})
    check(status == 422, f"entrée invalide refusée en 422 (reçu {status})")
    print("Smoke test réussi")


if __name__ == "__main__":
    main(sys.argv[1].rstrip("/"), sys.argv[2])
