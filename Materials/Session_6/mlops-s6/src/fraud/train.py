"""Train the model, log everything to MLflow and register a new version.

    python -m fraud.train

Promotion (the "staging" alias) is NOT done here: that is the job of the evaluation
gate (fraud.evaluate), as in the CI/CD pipeline.

The decision threshold is chosen on a validation slice and stored WITH the model
version (tag `decision_threshold`): a new model never runs with the previous
model's threshold.
"""

from __future__ import annotations

import hashlib
import json
import subprocess

import mlflow
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from fraud.config import MODEL_NAME, ROOT, TARGET, load_params
from fraud.features import SKOPS_TRUSTED_TYPES, build_pipeline, pick_threshold, select_features

REPORTS = ROOT / "reports"


def file_md5(path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def main() -> None:
    params = load_params()
    p_data, p_train = params["data"], params["train"]

    data_path = ROOT / p_data["train_path"]
    df = pd.read_csv(data_path).sort_values("time", kind="stable")
    # Chronological validation slice, like the train/golden split
    n_val = int(len(df) * p_train["validation_share"])
    fit, val = df.iloc[:-n_val], df.iloc[-n_val:]

    mlflow.set_experiment("fraud-detection")
    with mlflow.start_run() as run:
        # The three things to version: code, data, model
        mlflow.set_tags(
            {"git_commit": git_commit(), "data_md5": file_md5(data_path), "data_path": p_data["train_path"]}
        )
        mlflow.log_params({f"train.{k}": v for k, v in p_train.items()})
        mlflow.log_params({f"data.{k}": v for k, v in p_data.items()})

        pipeline = build_pipeline(p_train)
        pipeline.fit(select_features(fit), fit[TARGET])

        proba = pipeline.predict_proba(select_features(val))[:, 1]
        threshold = pick_threshold(val[TARGET], proba, p_train["min_recall"])
        pred = proba >= threshold
        metrics = {
            "val_auc": roc_auc_score(val[TARGET], proba),
            "val_avg_precision": average_precision_score(val[TARGET], proba),
            "val_precision": precision_score(val[TARGET], pred, zero_division=0),
            "val_recall": recall_score(val[TARGET], pred),
            "decision_threshold": threshold,
            "fraud_rate": float(df[TARGET].mean()),
        }
        mlflow.log_metrics(metrics)

        sample = select_features(val)
        model_info = mlflow.sklearn.log_model(
            pipeline,
            name="model",
            signature=infer_signature(sample, proba),
            input_example=sample.head(3),
            registered_model_name=MODEL_NAME,
            skops_trusted_types=SKOPS_TRUSTED_TYPES,
        )
        version = str(model_info.registered_model_version)
        MlflowClient().set_model_version_tag(MODEL_NAME, version, "decision_threshold", f"{threshold:.6f}")

        result = {"run_id": run.info.run_id, "model_name": MODEL_NAME, "model_version": version, **metrics}

    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "train.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
