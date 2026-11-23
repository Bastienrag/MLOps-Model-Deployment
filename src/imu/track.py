"""Entraîne le CNN et trace tout dans MLflow : paramètres, métriques, tags, modèle.

    python -m imu.track            # puis : python -m imu.register RUN_ID staging
"""

from pathlib import Path

import mlflow
import torch
import yaml
from mlflow.models import infer_signature

from imu.baseline import load_split
from imu.tracking import connect, data_fingerprints, git, git_dirty
from imu.train import predict, run


def flat(prefix, d):
    return {f"{prefix}.{k}": v for k, v in d.items()}


def run_metrics(results):
    """Métriques à journaliser : F1 macro et balanced accuracy, par trajet et par fenêtre, sur val et test."""
    out = {}
    for split in ("val", "test"):
        for level in ("ride", "window"):
            m = results[split][level]
            out[f"{split}_{level}_f1_macro"] = m["f1_macro"]
            out[f"{split}_{level}_balanced_accuracy"] = m["balanced_accuracy"]
    for cls, recall in results["test"]["ride"]["recall_per_class"].items():
        out[f"test_ride_recall_{cls.lower().replace(' ', '_')}"] = recall
    return out


def track(cfg):
    params, tp = cfg["data"], cfg["train"]
    uri = connect()
    with mlflow.start_run() as active:
        model, results = run(params, tp)
        mlflow.log_params({**flat("data", params), **flat("train", tp)})
        mlflow.log_metrics(run_metrics(results))
        for h in results["history"]:
            epoch_metrics = {"train_loss": h["train_loss"], "val_f1_macro_window": h["val_f1_macro_window"]}
            mlflow.log_metrics(epoch_metrics, step=h["epoch"])
        mlflow.set_tags({
            "git_commit": git("rev-parse", "HEAD"),
            "git_dirty": str(git_dirty()).lower(),
            **data_fingerprints(params),
            "best_epoch": str(results["best_epoch"]),
        })

        sample, _, _ = load_split(params["processed_dir"], "test")
        sample = sample[:8]
        # Le modèle servi renvoie des probabilités (softmax inclus), comme le dit sa signature.
        serving = torch.nn.Sequential(model.eval(), torch.nn.Softmax(dim=1))
        mlflow.pytorch.log_model(
            serving,
            name="model",
            signature=infer_signature(sample, predict(model, sample)),  # (n, 256, 6) -> (n, 5) probabilités
            input_example=sample[:3],
            code_paths=["src/imu"],  # le modèle se recharge sans le dépôt : la classe IMUNet voyage avec lui
        )
        # Ce que l'API devra connaître pour prétraiter : classes, taille de fenêtre, moyenne et écart-type.
        mlflow.log_artifact(str(Path(params["processed_dir"]) / "meta.json"))
        mlflow.log_artifact("params.yaml")
        mlflow.log_artifact("reports/baseline.json")
        print(f"MLflow {uri}")
        print(f"test trajets : F1 macro {results['test']['ride']['f1_macro']:.3f}, "
              f"balanced acc {results['test']['ride']['balanced_accuracy']:.3f}")
        print(f"run_id = {active.info.run_id}")
        return active.info.run_id


if __name__ == "__main__":
    track(yaml.safe_load(Path("params.yaml").read_text()))
