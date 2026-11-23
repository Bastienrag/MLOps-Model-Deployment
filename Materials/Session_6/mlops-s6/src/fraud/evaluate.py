"""Evaluation gate on the golden set: ROC AUC and PR AUC above thresholds, or fail.

    python -m fraud.evaluate            # evaluates the version produced by fraud.train
    python -m fraud.evaluate --no-promote

If the gate passes, the version gets the "staging" alias in the Model Registry.
Exit code 1 when the gate fails: the CI stops there.
"""

from __future__ import annotations

import argparse
import json
import sys

import mlflow
import pandas as pd
from mlflow import MlflowClient
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from fraud.config import ROOT, TARGET, load_params
from fraud.features import select_features

REPORTS = ROOT / "reports"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-promote", action="store_true")
    args = parser.parse_args()

    params = load_params()
    train = json.loads((REPORTS / "train.json").read_text())
    name, version, threshold = train["model_name"], train["model_version"], train["decision_threshold"]

    golden = pd.read_csv(ROOT / params["data"]["golden_path"])
    model = mlflow.sklearn.load_model(f"models:/{name}/{version}")
    proba = model.predict_proba(select_features(golden))[:, 1]
    y = golden[TARGET]

    gate = params["gate"]
    result = {
        "model_name": name,
        "model_version": version,
        "golden_auc": roc_auc_score(y, proba),
        "golden_avg_precision": average_precision_score(y, proba),
        "golden_precision": precision_score(y, proba >= threshold, zero_division=0),
        "golden_recall": recall_score(y, proba >= threshold),
        **{f"gate_{k}": v for k, v in gate.items()},
    }
    passed = result["golden_auc"] >= gate["min_auc"] and result["golden_avg_precision"] >= gate["min_avg_precision"]
    result["passed"] = passed
    (REPORTS / "eval.json").write_text(json.dumps(result, indent=2))

    client = MlflowClient()
    client.set_model_version_tag(name, version, "gate", "passed" if passed else "failed")
    with mlflow.start_run(run_id=train["run_id"]):
        mlflow.log_metrics({k: v for k, v in result.items() if k.startswith("golden_")})

    summary = f"ROC AUC {result['golden_auc']:.4f}, PR AUC {result['golden_avg_precision']:.4f}"
    if not passed:
        print(f"GATE FAILED: {summary} (min {gate['min_auc']} / {gate['min_avg_precision']})")
        return 1

    if not args.no_promote:
        client.set_registered_model_alias(name, "staging", version)
    print(f"GATE PASSED: {summary} -> {name} v{version} @staging")
    return 0


if __name__ == "__main__":
    sys.exit(main())
