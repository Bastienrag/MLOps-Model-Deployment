"""Métriques des 4 couches, cardinalité bornée, log JSON sans identifiant en clair, feedback."""

import json

import numpy as np
from prometheus_client import REGISTRY

from imu.api import monitoring


def value(name, **labels):
    return REGISTRY.get_sample_value(name, labels) or 0.0


def test_metrics_cover_the_four_layers(client, ride):
    before = value("imu_predictions_total", predicted_class="Safe overall")
    client.post("/v1/predict", json=ride)
    client.post("/v1/feedback", json={"predicted_class": "Safe overall", "true_class": "Tandem"})
    text = client.get("/metrics").text
    for name in (
        "imu_http_requests_total",            # 1. infrastructure
        "imu_http_request_duration_seconds_bucket",
        "imu_input_accel_std_bucket",          # 2. entrées
        "imu_input_samples_count",
        "imu_predictions_total",               # 3. prédictions
        "imu_prediction_confidence_bucket",
        "imu_feedback_total",                  # 4. résultat métier
    ):
        assert name in text
    assert value("imu_predictions_total", predicted_class="Safe overall") == before + 1


def test_correct_feedback_is_counted_by_true_class(client):
    before = value("imu_feedback_correct_total", true_class="Tandem")
    client.post("/v1/feedback", json={"predicted_class": "Tandem", "true_class": "Tandem"})
    client.post("/v1/feedback", json={"predicted_class": "Safe", "true_class": "Tandem"})
    assert value("imu_feedback_correct_total", true_class="Tandem") == before + 1


def test_route_label_is_the_route_not_the_url(client, ride):
    client.post("/v1/predict", json=ride)
    client.get("/chemin/invente/12345")
    client.get("/chemin/invente/67890")
    text = client.get("/metrics").text
    assert 'route="/v1/predict"' in text
    assert "12345" not in text and "invente" not in text
    assert value("imu_http_requests_total", method="GET", route="inconnue", status="404") >= 2


def test_errors_are_counted_by_status(client):
    before = value("imu_http_requests_total", method="POST", route="/v1/predict", status="422")
    client.post("/v1/predict", json={"samples": []})
    assert value("imu_http_requests_total", method="POST", route="/v1/predict", status="422") == before + 1


def test_metrics_endpoint_does_not_count_itself(client):
    client.get("/metrics")
    assert 'route="/metrics"' not in client.get("/metrics").text


def test_one_json_line_per_prediction_without_raw_id(client, ride, tmp_path, monkeypatch):
    log = tmp_path / "logs" / "predictions.jsonl"
    monkeypatch.setenv("PREDICTION_LOG", str(log))
    client.post("/v1/predict", json=ride)
    client.post("/v1/predict", json={**ride, "ride_id": None})
    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert [line["event"] for line in lines] == ["prediction", "prediction"]
    first = lines[0]
    assert first["model_version"] == "7" and first["predicted_class"] == "Safe overall"
    assert set(first["features"]) == {"n_samples", "accel_std", "gyro_rms", "missing_fraction"}
    assert first["features"]["n_samples"] == 600
    assert first["ride_hash"] and first["ride_hash"] != ride["ride_id"]
    assert ride["ride_id"] not in log.read_text()
    assert lines[1]["ride_hash"] is None


def test_feedback_is_logged_and_flags_correctness(client, tmp_path, monkeypatch):
    log = tmp_path / "events.jsonl"
    monkeypatch.setenv("PREDICTION_LOG", str(log))
    known = {"predicted_class": "Tandem", "true_class": "Tandem", "ride_id": "r-1"}
    ok = client.post("/v1/feedback", json=known)
    bad = client.post("/v1/feedback", json={"predicted_class": "Safe", "true_class": "Reckless"})
    assert ok.status_code == bad.status_code == 202
    assert ok.json() == {"correct": True} and bad.json() == {"correct": False}
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert [e["correct"] for e in events] == [True, False]
    assert "r-1" not in log.read_text()


def test_feedback_rejects_unknown_class(client):
    r = client.post("/v1/feedback", json={"predicted_class": "Safe", "true_class": "Inconnue"})
    assert r.status_code == 422


def test_hash_depends_on_salt_and_is_stable(monkeypatch):
    monkeypatch.setenv("LOG_HASH_SALT", "a")
    first = monitoring.hash_id("trajet-1")
    assert first == monitoring.hash_id("trajet-1")
    monkeypatch.setenv("LOG_HASH_SALT", "b")
    assert monitoring.hash_id("trajet-1") != first


def test_input_features_ignore_missing_rows():
    values = np.ones((10, 6))
    values[3] = np.nan
    f = monitoring.input_features(values)
    assert f["n_samples"] == 10 and f["missing_fraction"] == 0.1
    assert f["accel_std"] == 0.0 and np.isfinite(f["gyro_rms"])
