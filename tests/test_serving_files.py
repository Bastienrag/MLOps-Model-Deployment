"""L'image Docker n'a pas MLflow : l'API charge le modèle depuis models/cnn.pt et models/meta.json."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from imu.api import main

MODELS = Path("models")
EXAMPLES = Path("src/imu/web/static/examples")


def test_bundle_has_everything_the_service_needs():
    meta = json.loads((MODELS / "meta.json").read_text())
    assert len(meta["classes"]) == 5
    assert len(meta["mean"]) == len(meta["std"]) == 6
    assert meta["window"] == 256 and meta["stride"] == 128
    assert meta["channels"] and meta["dropout"] is not None


def test_load_from_files_predicts_probabilities(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "abc1234")
    loaded = main.load_from_files(MODELS)
    assert loaded.version == "abc1234"
    probs = loaded.model.predict(np.zeros((4, 256, 6), dtype=np.float32))
    assert probs.shape == (4, 5)
    assert probs.sum(axis=1) == pytest.approx(1.0, abs=1e-5)


def test_load_model_dispatches_on_the_uri(monkeypatch):
    monkeypatch.setattr(main, "load_from_registry", lambda uri: "registry")
    monkeypatch.setattr(main, "load_from_files", lambda uri: "files")
    assert main.load_model("models:/imu-classifier@production") == "registry"
    assert main.load_model("/app/models") == "files"


@pytest.mark.parametrize("example", sorted(p.name for p in EXAMPLES.glob("*.json") if p.name != "index.json"))
def test_api_serves_each_example_from_local_files(monkeypatch, example):
    monkeypatch.setattr(main, "MODEL_URI", str(MODELS))
    monkeypatch.setenv("MODEL_VERSION", "test-sha")
    ex = json.loads((EXAMPLES / example).read_text())
    samples = [dict(zip(ex["columns"], row, strict=True)) for row in ex["rows"]]
    with TestClient(main.app) as client:
        r = client.post("/v1/predict", json={"ride_id": ex["ride_id"], "samples": samples})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["predicted_class"] in json.loads((MODELS / "meta.json").read_text())["classes"]
        assert body["model_version"] == "test-sha"
        assert r.headers["X-Model-Version"] == "test-sha"
        assert client.get("/health").json()["model_loaded"] is True


def test_api_does_not_need_mlflow_nor_training_libraries():
    """Sinon l'image Docker, qui ne les installe pas, ne démarrerait pas."""
    code = (
        "import sys, imu.api.main;"
        "bad = [m for m in ('mlflow', 'sklearn', 'dvc') if m in sys.modules];"
        "assert not bad, bad"
    )
    run = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env={"PATH": "/usr/bin:/bin"}
    )
    assert run.returncode == 0, run.stderr
