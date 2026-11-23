"""Gate d'évaluation : refuse un modèle trop faible. Code de sortie 1 sous un seuil.

    python -m imu.gate

Évalue models/cnn.pt sur le jeu figé (ci/golden.npz), par trajet. Les seuils sont dans params.yaml
(`gate`) ; GATE_MIN_BALANCED_ACCURACY et GATE_MIN_F1_MACRO les remplacent, par exemple pour vérifier
qu'un échec met bien la pipeline au rouge. Le hasard donne 0,20 de balanced accuracy.
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

from imu.evaluate import aggregate_by_ride, metrics
from imu.model import IMUNet
from imu.train import predict


def thresholds(gate_params):
    return {
        "balanced_accuracy": float(
            os.getenv("GATE_MIN_BALANCED_ACCURACY") or gate_params["min_balanced_accuracy"]
        ),
        "f1_macro": float(os.getenv("GATE_MIN_F1_MACRO") or gate_params["min_f1_macro"]),
    }


def evaluate(cfg, model_path, golden_path):
    classes = cfg["data"]["classes"]
    tp = cfg["train"]
    model = IMUNet(6, len(classes), tp["channels"], tp["dropout"])
    model.load_state_dict(torch.load(model_path, weights_only=True))
    golden = np.load(golden_path)
    X = golden["X"].astype(np.float32)
    ride_probs, ride_y = aggregate_by_ride(predict(model, X), golden["y"], golden["ride_id"])
    result = metrics(ride_y, ride_probs.argmax(axis=1), classes)
    result["n_rides"], result["n_windows"] = int(len(ride_y)), int(len(X))
    return result


def check(result, limits):
    """Liste des seuils non atteints."""
    return [
        f"{name} {result[name]:.3f} < seuil {limit:.3f}"
        for name, limit in limits.items()
        if not result[name] >= limit  # `not >=` : un NaN échoue aussi
    ]


def main():
    cfg = yaml.safe_load(Path("params.yaml").read_text())
    limits = thresholds(cfg["gate"])
    result = evaluate(cfg, cfg["gate"]["model_path"], cfg["golden"]["path"])
    failures = check(result, limits)

    Path("reports").mkdir(exist_ok=True)
    report = {"passed": not failures, "thresholds": limits, "failures": failures, **result}
    Path("reports/gate.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    lines = [
        f"Gate d'évaluation : {'RÉUSSI' if not failures else 'ÉCHEC'}",
        f"  {result['n_rides']} trajets, {result['n_windows']} fenêtres (jeu figé)",
        f"  balanced accuracy {result['balanced_accuracy']:.3f} "
        f"(seuil {limits['balanced_accuracy']:.3f}, hasard 0.200)",
        f"  F1 macro          {result['f1_macro']:.3f} (seuil {limits['f1_macro']:.3f})",
        *[f"  ✗ {f}" for f in failures],
    ]
    print("\n".join(lines))
    if summary := os.getenv("GITHUB_STEP_SUMMARY"):
        Path(summary).open("a").write("### " + lines[0] + "\n```\n" + "\n".join(lines[1:]) + "\n```\n")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
