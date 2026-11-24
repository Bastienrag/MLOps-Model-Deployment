"""Découpe le temps d'une prédiction en postes, en mesurant chaque étape du code de l'API.

    uv run python loadtest/budget.py [répétitions]

Le réseau (aller-retour HTTP vers l'API) est mesuré à part, avec /health.
"""

import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

from imu.api.main import load_model
from imu.api.schemas import Ride
from imu.inference import aggregate, preprocess

EXAMPLES = Path("src/imu/web/static/examples")


def timed(fn, n):
    out = []
    for _ in range(n):
        t = time.perf_counter()
        result = fn()
        out.append((time.perf_counter() - t) * 1000)
    return statistics.median(out), result


def main(n):
    m = load_model("models")
    ex = json.loads((EXAMPLES / "reckless.json").read_text())
    body = {"ride_id": "x", "samples": [dict(zip(ex["columns"], r, strict=True)) for r in ex["rows"]]}
    raw = json.dumps(body)

    t_parse, _ = timed(lambda: json.loads(raw), n)
    t_valid, ride = timed(lambda: Ride.model_validate(body), n)
    t_pre, windows = timed(lambda: preprocess(ride.to_array(), m.mean, m.std, m.window, m.stride), n)
    t_inf, probs = timed(lambda: np.asarray(m.model.predict(windows)), n)
    t_post, _ = timed(lambda: aggregate(probs), n)
    rows = [("lecture JSON", t_parse), ("validation Pydantic", t_valid), ("prétraitement", t_pre),
            ("inférence", t_inf), ("post-traitement", t_post)]
    total = sum(t for _, t in rows)
    print(f"trajet de {len(ex['rows'])} mesures, {len(windows)} fenêtres, médiane sur {n} appels")
    for name, t in rows:
        print(f"{name:22s} {t:8.2f} ms  {100 * t / total:5.1f} %")
    print(f"{'total (hors réseau)':22s} {total:8.2f} ms")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 50)
