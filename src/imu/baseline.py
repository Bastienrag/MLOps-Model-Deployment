"""Baselines : classes de référence et régression logistique sur des statistiques de fenêtre.

    python -m imu.baseline
"""

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imu.evaluate import report


def load_split(processed_dir, name):
    d = np.load(Path(processed_dir) / f"{name}.npz")
    return d["X"], d["y"], d["ride_id"]


def window_features(X):
    """Moyenne, écart-type, min, max et variation moyenne de chaque capteur : 30 valeurs."""
    return np.concatenate(
        [
            X.mean(axis=1),
            X.std(axis=1),
            X.min(axis=1),
            X.max(axis=1),
            np.abs(np.diff(X, axis=1)).mean(axis=1),
        ],
        axis=1,
    )


def proba(model, X, n_classes):
    """predict_proba avec une colonne par classe, même si une classe manque à l'entraînement."""
    p = model.predict_proba(X)
    out = np.zeros((len(X), n_classes))
    out[:, model.classes_] = p
    return out


def run(params, baseline_params):
    processed = params["processed_dir"]
    classes = params["classes"]
    seed = baseline_params["seed"]
    Xtr, ytr, _ = load_split(processed, "train")
    Ftr = window_features(Xtr)

    models = {
        "majority": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                class_weight="balanced", max_iter=baseline_params["max_iter"], random_state=seed
            ),
        ),
    }
    results = {}
    for name, model in models.items():
        model.fit(Ftr, ytr)
        results[name] = {}
        for split in ("val", "test"):
            X, y, ride_id = load_split(processed, split)
            probs = proba(model, window_features(X), len(classes))
            results[name][split] = report(probs, y, ride_id, classes)
    return results


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("params.yaml").read_text())
    results = run(cfg["data"], cfg["baseline"])
    Path("reports").mkdir(exist_ok=True)
    Path("reports/baseline.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    for name, r in results.items():
        for split in ("val", "test"):
            m = r[split]["ride"]
            print(f"{name:9s} {split:4s} trajets : F1 macro {m['f1_macro']:.3f}  "
                  f"balanced acc {m['balanced_accuracy']:.3f}")
