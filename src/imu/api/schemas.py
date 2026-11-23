"""Contrat de l'API : Pydantic rejette toute entrée hors domaine par un 422, avant le modèle."""

from typing import Literal, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

MIN_SAMPLES = 256  # une fenêtre ; en dessous, aucune prédiction possible
MAX_SAMPLES = 10_000  # les trajets d'entraînement vont jusqu'à 4 767 points

ACC = {"ge": -50, "le": 50}  # m/s² : les données vont d'environ -20 à 13
GYRO = {"ge": -500, "le": 500}  # °/s : les données vont de -295 à 300


class Sample(BaseModel):
    """Une mesure de la centrale inertielle. Une ligne manquante est envoyée avec les 6 valeurs à null."""

    model_config = ConfigDict(extra="forbid")

    ax: float | None = Field(default=None, description="Accélération sur l'axe x (m/s²)", **ACC)
    ay: float | None = Field(default=None, description="Accélération sur l'axe y (m/s²)", **ACC)
    az: float | None = Field(default=None, description="Accélération sur l'axe z (m/s²)", **ACC)
    rx: float | None = Field(default=None, description="Vitesse angulaire autour de x (°/s)", **GYRO)
    ry: float | None = Field(default=None, description="Vitesse angulaire autour de y (°/s)", **GYRO)
    rz: float | None = Field(default=None, description="Vitesse angulaire autour de z (°/s)", **GYRO)

    @model_validator(mode="after")
    def all_or_nothing(self) -> Self:
        filled = [v is not None for v in (self.ax, self.ay, self.az, self.rx, self.ry, self.rz)]
        if any(filled) and not all(filled):
            raise ValueError("une mesure doit avoir ses 6 capteurs, ou aucun (ligne manquante)")
        return self


def example_ride(n=MIN_SAMPLES):
    """Trajet d'exemple valide pour Swagger : de quoi tester « Try it out » sans rien préparer."""
    t = np.arange(n) / 10
    rows = np.stack(
        [np.sin(t) * 0.5, -9.0 + np.sin(t / 2), np.cos(t) * 0.3, np.sin(t) * 2, np.cos(t) * 3, np.sin(t / 3)],
        axis=1,
    )
    return [
        dict(zip(("ax", "ay", "az", "rx", "ry", "rz"), map(float, np.round(r, 2)), strict=True)) for r in rows
    ]


class Ride(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        json_schema_extra={"examples": [{"ride_id": "trajet-001", "samples": example_ride()}]},
    )

    ride_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        description="Identifiant libre, renvoyé tel quel",
        examples=["trajet-001"],
    )
    samples: list[Sample] = Field(
        min_length=MIN_SAMPLES,
        max_length=MAX_SAMPLES,
        description=f"Mesures dans l'ordre du temps, au pas de 100 ms. Entre {MIN_SAMPLES} et {MAX_SAMPLES}.",
    )

    @model_validator(mode="after")
    def enough_measures(self) -> Self:
        valid = sum(s.ax is not None for s in self.samples)
        if valid < MIN_SAMPLES:
            raise ValueError(
                f"au moins {MIN_SAMPLES} mesures non manquantes sont nécessaires, {valid} reçues"
            )
        return self

    def to_array(self):
        """Tableau (temps, 6) avec NaN pour les lignes manquantes."""
        return np.array(
            [[s.ax, s.ay, s.az, s.rx, s.ry, s.rz] for s in self.samples], dtype=float
        )  # None devient NaN


class Prediction(BaseModel):
    ride_id: str | None = Field(description="Identifiant reçu dans la requête")
    predicted_class: str = Field(description="Catégorie du trajet", examples=["Safe"])
    probability: float = Field(ge=0, le=1, description="Probabilité de la catégorie prédite", examples=[0.62])
    probabilities: dict[str, float] = Field(description="Probabilité de chacune des 5 catégories")
    n_windows: int = Field(description="Nombre de fenêtres de 256 points moyennées")
    model_version: str = Field(description="Version du modèle dans le registry", examples=["3"])


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool
    model_name: str
    model_version: str = Field(description='"none" tant que le modèle n\'est pas chargé')
