"""Scoring API.

    MODEL_URI=model uvicorn fraud.api.main:app --port 8000
    MODEL_URI=models:/fraud-classifier@staging uvicorn fraud.api.main:app   # straight from the Registry

Endpoints: POST /predict, POST /feedback, GET /health, GET /metrics (Prometheus).

Decision threshold: DECISION_THRESHOLD if set, else the one chosen at training time
and exported with the model (model_meta.json), else 0.5.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from prometheus_client import make_asgi_app

from fraud.api import metrics as m
from fraud.api.schemas import Feedback, Health, Prediction, Transaction
from fraud.config import FEATURES

MODEL_URI = os.getenv("MODEL_URI", "model")
LOG_PATH = os.getenv("PREDICTION_LOG_PATH")  # e.g. logs/predictions.jsonl, read by fraud.drift

logger = logging.getLogger("fraud.api")
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))
logger.propagate = False

state: dict = {"model": None, "version": "none", "threshold": 0.5}


def redact(value: str | None) -> str | None:
    """Pseudonymise at log time: the clear value never leaves the API."""
    return hashlib.sha256(value.encode()).hexdigest()[:12] if value else None


def read_model_meta(uri: str) -> dict:
    meta = Path(uri) / "model_meta.json"
    return json.loads(meta.read_text()) if meta.exists() else {}


def log_event(event: dict) -> None:
    line = json.dumps(event, default=str)
    logger.info(line)
    if LOG_PATH:
        Path(LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(LOG_PATH, "a") as f:
            f.write(line + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        state["model"] = mlflow.sklearn.load_model(MODEL_URI)
        meta = read_model_meta(MODEL_URI)
        state["version"] = str(meta.get("version", os.getenv("MODEL_VERSION", MODEL_URI)))
        state["threshold"] = float(os.getenv("DECISION_THRESHOLD", meta.get("decision_threshold", 0.5)))
        m.MODEL_INFO.labels(version=state["version"]).set(1)
    except Exception as exc:  # the API still starts, /health reports "degraded"
        logger.error(json.dumps({"event": "model_load_failed", "model_uri": MODEL_URI, "error": str(exc)}))
    yield


app = FastAPI(title="Fraud scoring API", version="0.2.0", lifespan=lifespan)
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def observe(request: Request, call_next):
    start = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        path = request.url.path
        if not path.startswith("/metrics"):
            m.REQUESTS.labels(endpoint=path, status=str(status)).inc()
            m.LATENCY.labels(endpoint=path).observe(time.perf_counter() - start)


@app.get("/health", response_model=Health)
def health() -> Health:
    loaded = state["model"] is not None
    return Health(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model_version=state["version"],
        decision_threshold=state["threshold"],
    )


@app.post("/predict", response_model=Prediction)
def predict(tx: Transaction) -> Prediction:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.perf_counter()
    features = tx.model_dump(include=set(FEATURES))
    proba = float(state["model"].predict_proba(pd.DataFrame([features], columns=FEATURES))[0, 1])
    is_fraud = proba >= state["threshold"]

    m.INPUT_AMOUNT.observe(tx.amount)
    for component in m.MONITORED_COMPONENTS:
        m.INPUT_COMPONENT.labels(component=component).observe(features[component])
    m.PREDICTION_PROBA.observe(proba)
    if is_fraud:
        m.PREDICTED_FRAUD.inc()

    log_event(
        {
            "ts": datetime.now(UTC).isoformat(),
            "event": "prediction",
            "transaction_id": tx.transaction_id,
            "customer_hash": redact(tx.customer_id),
            **features,
            "fraud_probability": round(proba, 6),
            "is_fraud": is_fraud,
            "model_version": state["version"],
            "inference_ms": round((time.perf_counter() - start) * 1000, 2),
        }
    )
    return Prediction(
        transaction_id=tx.transaction_id, fraud_probability=proba, is_fraud=is_fraud, model_version=state["version"]
    )


@app.post("/feedback", status_code=204)
def feedback(fb: Feedback) -> None:
    outcome = {(True, True): "tp", (True, False): "fp", (False, True): "fn", (False, False): "tn"}
    m.FEEDBACK.labels(outcome=outcome[(fb.predicted_fraud, fb.actual_fraud)]).inc()
    log_event({"ts": datetime.now(UTC).isoformat(), "event": "feedback", **fb.model_dump()})
