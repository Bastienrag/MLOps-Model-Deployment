"""API de prédiction : classe un trajet IMU dans l'une des 5 catégories.

    MLFLOW_PORT=5001 uv run uvicorn imu.api.main:app --port 8000
    MODEL_URI=models:/imu-classifier@staging uv run uvicorn imu.api.main:app    # autre alias

Routes : POST /v1/predict, GET /health. Documentation : /docs (Swagger).
Le modèle est chargé une fois au démarrage depuis le registry MLflow, par nom et alias.
"""

import json
import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

import mlflow
import numpy as np
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from mlflow import MlflowClient

from imu.api.schemas import Health, Prediction, Ride
from imu.inference import aggregate, preprocess
from imu.tracking import MODEL_NAME, tracking_uri

MODEL_URI = os.getenv("MODEL_URI", f"models:/{MODEL_NAME}@production")
log = logging.getLogger("uvicorn.error")


@dataclass
class Loaded:
    model: object
    version: str
    classes: list[str]
    mean: np.ndarray
    std: np.ndarray
    window: int
    stride: int


state: dict = {"loaded": None}


def load_from_registry(uri):
    """Charge le modèle et les paramètres de prétraitement qui l'accompagnent (meta.json du même run)."""
    mlflow.set_tracking_uri(tracking_uri())
    name, _, alias = uri.removeprefix("models:/").partition("@")
    version = MlflowClient().get_model_version_by_alias(name, alias)
    meta_path = mlflow.artifacts.download_artifacts(run_id=version.run_id, artifact_path="meta.json")
    with open(meta_path) as f:
        meta = json.load(f)
    return Loaded(
        model=mlflow.pyfunc.load_model(uri),
        version=str(version.version),
        classes=meta["classes"],
        mean=np.array(meta["mean"], dtype=np.float32),
        std=np.array(meta["std"], dtype=np.float32),
        window=meta["window"],
        stride=meta["stride"],
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Au démarrage, une seule fois. Sans modèle le service démarre quand même :
    /health et /v1/predict répondent alors 503."""
    try:
        state["loaded"] = load_from_registry(MODEL_URI)
        log.info("modèle %s chargé (version %s)", MODEL_URI, state["loaded"].version)
    except Exception as exc:
        state["loaded"] = None
        log.error("modèle %s non chargé : %s", MODEL_URI, exc)
    yield
    state["loaded"] = None


app = FastAPI(
    title="API de classification de trajets IMU",
    description=(
        "Reçoit les mesures d'une centrale inertielle (accéléromètre et gyroscope, 10 Hz) d'un trajet "
        "et renvoie sa catégorie : **Safe**, **Safe overall**, **Bad weather**, **Tandem** ou "
        "**Reckless**.\n\n"
        "Le trajet est découpé en fenêtres de 256 points ; la prédiction est la moyenne de leurs "
        "probabilités. Les classes sont très déséquilibrées : se fier au vecteur `probabilities`, "
        "pas seulement à la classe."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def model_version_header(request: Request, call_next):
    """Chaque réponse, erreurs comprises, indique la version du modèle qui était chargée."""
    response = await call_next(request)
    loaded = state["loaded"]
    response.headers["X-Model-Version"] = loaded.version if loaded else "none"
    return response


def model_or_503():
    if state["loaded"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return state["loaded"]


@app.get(
    "/health",
    response_model=Health,
    summary="État du service",
    responses={503: {"model": Health, "description": "Le modèle n'est pas chargé"}},
)
def health():
    loaded = state["loaded"]
    body = Health(
        status="ok" if loaded else "unavailable",
        model_loaded=loaded is not None,
        model_name=MODEL_NAME,
        model_version=loaded.version if loaded else "none",
    )
    return JSONResponse(body.model_dump(), status_code=200 if loaded else 503)


v1 = APIRouter(prefix="/v1", tags=["Prédiction"])


@v1.post(
    "/predict",
    response_model=Prediction,
    summary="Classe un trajet",
    responses={
        422: {"description": "Entrée invalide : la réponse détaille chaque champ en cause"},
        503: {"description": "Le modèle n'est pas chargé"},
    },
)
def predict(ride: Ride) -> Prediction:
    m = model_or_503()
    windows = preprocess(ride.to_array(), m.mean, m.std, m.window, m.stride)
    probs = aggregate(np.asarray(m.model.predict(windows)))
    best = int(probs.argmax())
    return Prediction(
        ride_id=ride.ride_id,
        predicted_class=m.classes[best],
        probability=float(probs[best]),
        probabilities={c: float(p) for c, p in zip(m.classes, probs, strict=True)},
        n_windows=len(windows),
        model_version=m.version,
    )


app.include_router(v1)
