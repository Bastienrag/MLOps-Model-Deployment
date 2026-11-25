"""Simulateur de trafic : envoie à l'API des trajets réalistes, puis la vérité terrain de la plupart.

    python monitoring/simulate_traffic.py --url http://localhost:8000 --rides 300 --rate 5 --seed 0

Chaque trajet est un des 5 exemples réels (un par classe), tiré dans les proportions du jeu GBG500 (83 %
de Safe), recadré à une longueur aléatoire, légèrement bruité et mis à l'échelle, avec quelques lignes
manquantes. La graine fixe tout : deux lancements envoient exactement les mêmes trajets, ce qui donne une
baseline reproductible pour les tableaux de bord, les alertes et la détection de dérive.

Bibliothèque standard seulement.
"""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path

EXAMPLES = Path("src/imu/web/static/examples")
PROPORTIONS = {"Safe": 0.832, "Safe overall": 0.076, "Bad weather": 0.044, "Tandem": 0.036, "Reckless": 0.012}


def load_examples():
    out = {}
    for entry in json.loads((EXAMPLES / "index.json").read_text()):
        ex = json.loads((EXAMPLES / entry["file"]).read_text())
        out[entry["true_class"]] = (ex["columns"], ex["rows"])
    return out


def make_ride(rng, columns, rows):
    """Variante d'un trajet réel : recadrage, mise à l'échelle par capteur, bruit, lignes manquantes."""
    length = rng.randint(300, len(rows))
    start = rng.randint(0, len(rows) - length)
    scale = [rng.gauss(1, 0.05) for _ in columns]
    samples = []
    for row in rows[start:start + length]:
        if rng.random() < 0.01 or any(v is None for v in row):
            samples.append(dict.fromkeys(columns))  # ligne manquante : les 6 capteurs à null
        else:
            noisy = (round(v * k + rng.gauss(0, 0.05), 3) for v, k in zip(row, scale, strict=True))
            samples.append(dict(zip(columns, noisy, strict=True)))
    return samples


def post(url, payload):
    request = urllib.request.Request(url, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as err:
        return err.code, {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--rides", type=int, default=200)
    ap.add_argument("--rate", type=float, default=5, help="trajets par seconde (0 = au plus vite)")
    ap.add_argument("--feedback", type=float, default=0.7, help="part des trajets avec vérité terrain")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    examples = load_examples()
    classes = list(PROPORTIONS)
    counts = dict.fromkeys(classes, 0)
    correct = errors = feedback = 0
    for i in range(args.rides):
        true_class = rng.choices(classes, [PROPORTIONS[c] for c in classes])[0]
        samples = make_ride(rng, *examples[true_class])
        status, body = post(f"{args.url}/v1/predict", {"ride_id": f"sim-{args.seed}-{i}", "samples": samples})
        if status != 200:
            errors += 1
            continue
        counts[body["predicted_class"]] += 1
        if rng.random() < args.feedback:
            truth = {"ride_id": f"sim-{args.seed}-{i}", "true_class": true_class,
                     "predicted_class": body["predicted_class"]}
            post(f"{args.url}/v1/feedback", truth)
            feedback += 1
            correct += body["predicted_class"] == true_class
        if args.rate:
            time.sleep(1 / args.rate)
    print(f"{args.rides} trajets, {errors} erreurs, {feedback} vérités terrain ({correct} justes)")
    print("classes prédites :", counts)


if __name__ == "__main__":
    main()
