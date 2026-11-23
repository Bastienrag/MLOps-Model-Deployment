"""Test à 1e-6 : réentraîne depuis le checkout courant et compare avec ce que MLflow a gardé.

    python -m imu.repro RUN_ID

Les paramètres viennent du run, pas de params.yaml : on veut ce qui a vraiment été utilisé.
"""

import ast
import sys

import mlflow

from imu.track import run_metrics
from imu.tracking import connect, data_fingerprints, git
from imu.train import run


def typed(value):
    """Les paramètres reviennent de MLflow en texte : retrouve les nombres et les listes."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return value


if __name__ == "__main__":
    connect()
    run_data = mlflow.get_run(sys.argv[1]).data
    tags, logged = run_data.tags, run_data.metrics
    tp = {k.removeprefix("train."): typed(v) for k, v in run_data.params.items() if k.startswith("train.")}
    data = {k.removeprefix("data."): typed(v) for k, v in run_data.params.items() if k.startswith("data.")}

    checks = {"code  (git commit)": git("rev-parse", "HEAD") == tags["git_commit"]}
    fingerprints = data_fingerprints(data)
    for name, value in fingerprints.items():
        checks[f"{name:<19}"] = value == tags[name]

    _, results = run(data, tp)
    metrics = run_metrics(results)
    for name in logged:
        if name in metrics:  # les métriques par époque (train_loss...) ne sont pas dans `metrics`
            delta = abs(metrics[name] - logged[name])
            checks[f"{name:<34} delta={delta:.1e}"] = delta < 1e-6

    for label, ok in checks.items():
        print("PASS" if ok else "FAIL", label)
    print("\nREPRODUIT" if all(checks.values()) else "\nNON REPRODUCTIBLE")
    sys.exit(not all(checks.values()))
