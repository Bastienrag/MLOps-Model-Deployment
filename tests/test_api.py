import pytest

from tests.conftest import CLASSES, PROBS


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {
        "status": "ok",
        "model_loaded": True,
        "model_name": "imu-classifier",
        "model_version": "7",
    }


def test_predict_nominal(client, ride):
    r = client.post("/v1/predict", json=ride)
    assert r.status_code == 200
    body = r.json()
    assert body["predicted_class"] == "Safe overall"  # argmax de PROBS
    assert body["probability"] == pytest.approx(0.5)
    assert body["probabilities"] == pytest.approx(dict(zip(CLASSES, PROBS, strict=True)))
    assert sum(body["probabilities"].values()) == pytest.approx(1.0)
    assert body["n_windows"] == 3  # (600 - 256) // 128 + 1
    assert body["ride_id"] == "trajet-test"


def test_every_response_carries_the_model_version(client, ride):
    ok = client.post("/v1/predict", json=ride)
    assert ok.json()["model_version"] == "7"
    assert ok.headers["X-Model-Version"] == "7"
    # aussi sur les erreurs et sur /health
    assert client.post("/v1/predict", json={}).headers["X-Model-Version"] == "7"
    assert client.get("/health").headers["X-Model-Version"] == "7"


def test_missing_rows_are_accepted_and_interpolated(client, ride):
    blank = dict.fromkeys(("ax", "ay", "az", "rx", "ry", "rz"))
    ride["samples"][100:103] = [blank] * 3
    assert client.post("/v1/predict", json=ride).status_code == 200


# --- 422 : l'entrée invalide est rejetée avec le détail des champs -----------------------------


def errors(response):
    assert response.status_code == 422
    return response.json()["detail"]


def test_too_short_ride_is_rejected(client, ride):
    ride["samples"] = ride["samples"][:255]
    detail = errors(client.post("/v1/predict", json=ride))
    assert detail[0]["loc"] == ["body", "samples"]


def test_out_of_range_sensor_names_the_field(client, ride):
    ride["samples"][5]["ry"] = 5000
    detail = errors(client.post("/v1/predict", json=ride))
    assert detail[0]["loc"] == ["body", "samples", 5, "ry"]


def test_wrong_type_is_rejected(client, ride):
    ride["samples"][0]["ax"] = "beaucoup"
    detail = errors(client.post("/v1/predict", json=ride))
    assert detail[0]["loc"] == ["body", "samples", 0, "ax"]


def test_unknown_field_is_rejected(client, ride):
    ride["samples"][0]["bx"] = 1.0
    detail = errors(client.post("/v1/predict", json=ride))
    assert detail[0]["type"] == "extra_forbidden"
    assert detail[0]["loc"] == ["body", "samples", 0, "bx"]


def test_unknown_top_level_field_is_rejected(client, ride):
    assert errors(client.post("/v1/predict", json={**ride, "vitesse": 3}))[0]["type"] == "extra_forbidden"


def test_partially_missing_row_is_rejected(client, ride):
    ride["samples"][7]["az"] = None
    detail = errors(client.post("/v1/predict", json=ride))
    assert "6 capteurs" in detail[0]["msg"]


def test_ride_with_too_many_missing_rows_is_rejected(client, ride):
    blank = dict.fromkeys(("ax", "ay", "az", "rx", "ry", "rz"))
    ride["samples"] = [blank] * 300 + ride["samples"][:200]
    detail = errors(client.post("/v1/predict", json=ride))
    assert "mesures non manquantes" in detail[0]["msg"]


def test_missing_samples_is_rejected(client):
    detail = errors(client.post("/v1/predict", json={"ride_id": "x"}))
    assert detail[0]["loc"] == ["body", "samples"]


def test_empty_ride_id_is_rejected(client, ride):
    ride["ride_id"] = ""
    assert errors(client.post("/v1/predict", json=ride))[0]["loc"] == ["body", "ride_id"]


# --- 503 : le modèle n'est pas chargé ----------------------------------------------------------


def test_health_is_503_without_model(client_without_model):
    r = client_without_model.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "unavailable"
    assert r.json()["model_loaded"] is False
    assert r.json()["model_version"] == "none"


def test_predict_is_503_without_model(client_without_model, ride):
    r = client_without_model.post("/v1/predict", json=ride)
    assert r.status_code == 503
    assert r.headers["X-Model-Version"] == "none"


def test_invalid_input_is_still_422_without_model(client_without_model):
    assert client_without_model.post("/v1/predict", json={}).status_code == 422


# --- Swagger ------------------------------------------------------------------------------------


def test_swagger_page_is_served(client):
    assert client.get("/docs").status_code == 200


def test_openapi_documents_the_contract(client):
    spec = client.get("/openapi.json").json()
    predict = spec["paths"]["/v1/predict"]["post"]
    assert predict["summary"]
    assert {"200", "422", "503"} <= set(predict["responses"])
    schemas = spec["components"]["schemas"]
    # chaque champ est décrit, et l'exemple de requête est valide : « Try it out » fonctionne
    assert all("description" in p for p in schemas["Sample"]["properties"].values())
    assert all("description" in p for p in schemas["Prediction"]["properties"].values())
    example = schemas["Ride"]["examples"][0]
    assert len(example["samples"]) == 256


def test_swagger_example_is_a_valid_request(client):
    example = client.get("/openapi.json").json()["components"]["schemas"]["Ride"]["examples"][0]
    assert client.post("/v1/predict", json=example).status_code == 200
