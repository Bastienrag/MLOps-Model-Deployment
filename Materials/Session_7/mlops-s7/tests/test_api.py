import importlib
import json

import pytest
from fastapi.testclient import TestClient

VALID = {"transaction_id": "tx-1", "customer_id": "12345", "amount": 42.5} | {f"v{i}": 0.1 for i in range(1, 29)}


@pytest.fixture
def client(model_dir, tmp_path, monkeypatch):
    monkeypatch.setenv("MODEL_URI", str(model_dir))
    monkeypatch.setenv("PREDICTION_LOG_PATH", str(tmp_path / "predictions.jsonl"))
    monkeypatch.delenv("DECISION_THRESHOLD", raising=False)
    import fraud.api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        c.log_path = tmp_path / "predictions.jsonl"
        yield c


def test_health_exposes_model_and_threshold(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model_loaded"]
    assert body["model_version"] == "test" and body["decision_threshold"] == 0.3


def test_predict(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    assert 0 <= r.json()["fraud_probability"] <= 1


def test_customer_id_is_optional(client):
    payload = {k: v for k, v in VALID.items() if k != "customer_id"}
    assert client.post("/predict", json=payload).status_code == 200


@pytest.mark.parametrize("field,value", [("amount", -5), ("v14", 1e6), ("v3", "abc")])
def test_invalid_input_is_rejected(client, field, value):
    assert client.post("/predict", json={**VALID, field: value}).status_code == 422


def test_missing_component_is_rejected(client):
    payload = {k: v for k, v in VALID.items() if k != "v28"}
    assert client.post("/predict", json=payload).status_code == 422


def test_unknown_field_is_rejected(client):
    assert client.post("/predict", json={**VALID, "email": "a@b.c"}).status_code == 422


def test_customer_id_is_redacted_in_logs(client):
    client.post("/predict", json=VALID)
    line = json.loads(client.log_path.read_text().splitlines()[-1])
    assert "customer_id" not in line and VALID["customer_id"] not in json.dumps(line)
    assert line["customer_hash"]


def test_metrics_exposed(client):
    client.post("/predict", json=VALID)
    text = client.get("/metrics/").text
    assert "api_request_duration_seconds" in text and "input_component_value" in text


def test_feedback(client):
    r = client.post("/feedback", json={"transaction_id": "tx-1", "predicted_fraud": False, "actual_fraud": True})
    assert r.status_code == 204
