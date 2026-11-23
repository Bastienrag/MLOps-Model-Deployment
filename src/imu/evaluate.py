"""Métriques adaptées au déséquilibre des classes, par fenêtre et par trajet."""

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, recall_score


def metrics(y_true, y_pred, classes):
    labels = list(range(len(classes)))
    recalls = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "f1_macro": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float((np.asarray(y_true) == np.asarray(y_pred)).mean()),
        "recall_per_class": {c: float(r) for c, r in zip(classes, recalls, strict=True)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def aggregate_by_ride(probs, y, ride_id):
    """Moyenne des probabilités des fenêtres de chaque trajet. Renvoie (probs, labels) par trajet."""
    rides = np.unique(ride_id)
    ride_probs = np.stack([probs[ride_id == r].mean(axis=0) for r in rides])
    ride_y = np.array([y[ride_id == r][0] for r in rides])
    return ride_probs, ride_y


def report(probs, y, ride_id, classes):
    """Métriques par fenêtre et par trajet (le trajet est l'unité à classer)."""
    ride_probs, ride_y = aggregate_by_ride(probs, y, ride_id)
    return {
        "window": metrics(y, probs.argmax(axis=1), classes),
        "ride": metrics(ride_y, ride_probs.argmax(axis=1), classes),
    }
