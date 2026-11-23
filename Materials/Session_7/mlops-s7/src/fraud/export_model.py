"""Download a Model Registry version into ./model, to bake it into the Docker image.

    python -m fraud.export_model                 # staging alias
    python -m fraud.export_model --alias champion

The API image contains the model: a tagged image is a frozen model, and the
Kubernetes pod never needs to reach the MLflow server.
"""

from __future__ import annotations

import argparse
import json
import shutil

import mlflow
from mlflow import MlflowClient

from fraud.config import MODEL_NAME, ROOT


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--alias", default="staging")
    parser.add_argument("--out", default=str(ROOT / "model"))
    args = parser.parse_args()

    mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, args.alias)
    shutil.rmtree(args.out, ignore_errors=True)
    mlflow.artifacts.download_artifacts(artifact_uri=f"models:/{MODEL_NAME}/{mv.version}", dst_path=args.out)
    meta = {
        "name": MODEL_NAME,
        "version": mv.version,
        "alias": args.alias,
        "run_id": mv.run_id,
        "decision_threshold": float(mv.tags.get("decision_threshold", 0.5)),
    }
    with open(f"{args.out}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"{MODEL_NAME} v{mv.version} (@{args.alias}, threshold {meta['decision_threshold']:.4f}) -> {args.out}")


if __name__ == "__main__":
    main()
