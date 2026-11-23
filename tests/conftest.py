import numpy as np
import pytest
from fastapi.testclient import TestClient

from imu.api import main
from imu.api.schemas import example_ride

CLASSES = ["Safe", "Safe overall", "Bad weather", "Tandem", "Reckless"]
PROBS = np.array([0.1, 0.5, 0.2, 0.15, 0.05])


class StubModel:
    """Même interface que le modèle MLflow : fenêtres (n, 256, 6) -> probabilités (n, 5)."""

    def predict(self, windows):
        assert windows.shape[1:] == (256, 6)
        return np.tile(PROBS, (len(windows), 1))


def stub_loader(uri):
    return main.Loaded(
        model=StubModel(),
        version="7",
        classes=CLASSES,
        mean=np.zeros(6, dtype=np.float32),
        std=np.ones(6, dtype=np.float32),
        window=256,
        stride=128,
    )


def failing_loader(uri):
    raise ConnectionError("registry injoignable")


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(main, "load_from_registry", stub_loader)
    with TestClient(main.app) as c:  # le `with` exécute le démarrage : le modèle est chargé
        yield c


@pytest.fixture
def client_without_model(monkeypatch):
    monkeypatch.setattr(main, "load_from_registry", failing_loader)
    with TestClient(main.app) as c:
        yield c


@pytest.fixture
def ride():
    """Trajet valide de 600 mesures : 3 fenêtres de 256 points au pas de 128."""
    return {"ride_id": "trajet-test", "samples": example_ride(600)}
