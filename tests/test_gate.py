import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

from imu.gate import check, evaluate, thresholds
from imu.model import IMUNet

CFG = yaml.safe_load(Path("params.yaml").read_text())
LIMITS = thresholds(CFG["gate"])


def test_check_lists_every_threshold_missed():
    result = {"balanced_accuracy": 0.25, "f1_macro": 0.2}
    assert len(check(result, {"balanced_accuracy": 0.3, "f1_macro": 0.25})) == 2
    assert check(result, {"balanced_accuracy": 0.2, "f1_macro": 0.2}) == []


def test_check_fails_on_nan():
    assert check(
        {"balanced_accuracy": float("nan"), "f1_macro": 0.5}, {"balanced_accuracy": 0.3, "f1_macro": 0.2}
    )


def test_thresholds_come_from_params_and_can_be_overridden(monkeypatch):
    monkeypatch.delenv("GATE_MIN_BALANCED_ACCURACY", raising=False)
    assert thresholds(CFG["gate"])["balanced_accuracy"] == CFG["gate"]["min_balanced_accuracy"]
    monkeypatch.setenv("GATE_MIN_BALANCED_ACCURACY", "0.99")
    assert thresholds(CFG["gate"])["balanced_accuracy"] == 0.99


def test_thresholds_are_above_chance():
    assert LIMITS["balanced_accuracy"] > 1 / len(CFG["data"]["classes"])


def test_committed_model_passes_the_gate():
    result = evaluate(CFG, CFG["gate"]["model_path"], CFG["golden"]["path"])
    assert check(result, LIMITS) == []
    assert result["n_rides"] == 76


def test_untrained_model_is_refused(tmp_path):
    torch.manual_seed(0)
    tp = CFG["train"]
    untrained = IMUNet(6, len(CFG["data"]["classes"]), tp["channels"], tp["dropout"])
    path = tmp_path / "untrained.pt"
    torch.save(untrained.state_dict(), path)
    result = evaluate(CFG, path, CFG["golden"]["path"])
    assert check(result, LIMITS), (
        f"un modèle non entraîné ne doit pas passer : {result['balanced_accuracy']:.3f}"
    )


def test_gate_command_exits_with_an_error_under_the_threshold(tmp_path):
    env = {"GATE_MIN_BALANCED_ACCURACY": "0.99", "PATH": "/usr/bin:/bin", "PYTHONPATH": "src"}
    run = subprocess.run([sys.executable, "-m", "imu.gate"], capture_output=True, text=True, env=env)
    assert run.returncode == 1
    assert "ÉCHEC" in run.stdout


@pytest.mark.parametrize("name", ["ci/golden.npz", "models/cnn.pt"])
def test_gate_files_are_small_enough_for_git(name):
    assert Path(name).stat().st_size < 5_000_000
