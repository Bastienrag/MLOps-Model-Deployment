"""Seuils d'évaluation appliqués avant l'enregistrement d'un modèle (étape « evaluate » du DAG Airflow)."""

from types import SimpleNamespace

import mlflow
import pytest

from imu.check_run import check

GATE = {"min_balanced_accuracy": 0.30, "min_f1_macro": 0.25}


def fake_run(monkeypatch, bacc, f1):
    metrics = {"test_ride_balanced_accuracy": bacc, "test_ride_f1_macro": f1}
    run = SimpleNamespace(data=SimpleNamespace(metrics=metrics))
    monkeypatch.setattr(mlflow, "get_run", lambda run_id: run)


@pytest.mark.parametrize(("bacc", "f1", "expected"), [
    (0.55, 0.43, True),
    (0.30, 0.25, True),    # exactement au seuil : accepté
    (0.29, 0.43, False),   # balanced accuracy trop basse
    (0.55, 0.20, False),   # F1 macro trop bas
])
def test_check_applies_both_thresholds(monkeypatch, bacc, f1, expected):
    fake_run(monkeypatch, bacc, f1)
    assert check("run", GATE) is expected
