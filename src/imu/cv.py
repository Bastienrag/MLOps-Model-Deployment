"""Validation croisée stratifiée par trajet sur train+val (les trajets de test ne sont pas touchés).

    python -m imu.cv

Les prédictions des trajets mis de côté de chaque pli sont rassemblées : les métriques portent
sur tous les trajets (424, dont 5 Reckless), bien plus fiables que sur le seul jeu de validation.
"""

import json
from pathlib import Path

import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from imu.baseline import load_split, window_features
from imu.evaluate import metrics
from imu.train import fit, predict


def pooled(params):
    """Fenêtres de train et val réunies, et label par trajet."""
    parts = [load_split(params["processed_dir"], s) for s in ("train", "val")]
    X, y, ride_id = (np.concatenate(p) for p in zip(*parts, strict=True))
    rides = np.unique(ride_id)
    ride_y = np.array([y[ride_id == r][0] for r in rides])
    return X, y, ride_id, rides, ride_y


def ride_mean(probs, ride_id, rides):
    return np.stack([probs[ride_id == r].mean(axis=0) for r in rides])


def folds(rides, ride_y, n_splits, seed):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(rides, ride_y))


def run_logreg(params, cv):
    classes = params["classes"]
    X, y, ride_id, rides, ride_y = pooled(params)
    F = window_features(X)
    oof = np.zeros((len(rides), len(classes)))
    for tr, te in folds(rides, ride_y, cv["folds"], cv["seed"]):
        in_tr = np.isin(ride_id, rides[tr])
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=cv["seed"]),
        )
        model.fit(F[in_tr], y[in_tr])
        in_te = np.isin(ride_id, rides[te])
        p = np.zeros((in_te.sum(), len(classes)))
        p[:, model.classes_] = model.predict_proba(F[in_te])
        oof[te] = ride_mean(p, ride_id[in_te], rides[te])
    return metrics(ride_y, oof.argmax(axis=1), classes)


def run_cnn(params, tp, cv):
    """Renvoie les métriques par trajet (prédictions hors pli) à chaque époque."""
    classes = params["classes"]
    X, y, ride_id, rides, ride_y = pooled(params)
    tp = {**tp, "epochs": cv["epochs"]}
    oof = np.zeros((tp["epochs"], len(rides), len(classes)))
    for k, (tr, te) in enumerate(folds(rides, ride_y, cv["folds"], cv["seed"]), start=1):
        in_tr, in_te = np.isin(ride_id, rides[tr]), np.isin(ride_id, rides[te])

        def on_epoch(model, epoch, loss, in_te=in_te, te=te, k=k):
            p = predict(model, X[in_te])
            oof[epoch - 1, te] = ride_mean(p, ride_id[in_te], rides[te])
            print(f"pli {k}/{cv['folds']} époque {epoch:2d} perte {loss:.4f}", flush=True)

        fit(X[in_tr], y[in_tr], tp, len(classes), on_epoch)
    curve = [metrics(ride_y, oof[e].argmax(axis=1), classes) for e in range(tp["epochs"])]
    best = int(np.argmax([m["f1_macro"] for m in curve]))
    return {
        "best_epoch": best + 1,
        "best": curve[best],
        "curve": [{"epoch": e + 1, "f1_macro": m["f1_macro"], "balanced_accuracy": m["balanced_accuracy"]}
                  for e, m in enumerate(curve)],
    }


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("params.yaml").read_text())
    Path("reports").mkdir(exist_ok=True)
    out = {"logreg": run_logreg(cfg["data"], cfg["cv"]), "cnn": run_cnn(cfg["data"], cfg["train"], cfg["cv"])}
    Path("reports/cv.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    lr, cnn = out["logreg"], out["cnn"]["best"]
    print(f"logreg : F1 macro {lr['f1_macro']:.3f}  balanced acc {lr['balanced_accuracy']:.3f}")
    print(f"cnn    : F1 macro {cnn['f1_macro']:.3f}  balanced acc {cnn['balanced_accuracy']:.3f} "
          f"(époque {out['cnn']['best_epoch']})")
