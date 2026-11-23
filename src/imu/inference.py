"""Du trajet brut aux probabilités de classe : le même prétraitement qu'à l'entraînement."""

import numpy as np
import pandas as pd

from imu.data import SENSORS, clean_ride, windows


def preprocess(values, mean, std, window, stride):
    """Trajet (temps, 6 capteurs), avec NaN pour les mesures manquantes -> fenêtres normalisées.

    Réutilise `clean_ride` (interpolation des lignes manquantes) et `windows` de l'entraînement, avec
    les statistiques de normalisation du train enregistrées avec le modèle.
    """
    ride = pd.DataFrame(values, columns=SENSORS)
    ride.insert(0, "time_index", np.arange(len(ride)) * 100)
    series = clean_ride(ride)
    return ((windows(series, window, stride) - mean) / std).astype(np.float32)


def aggregate(window_probs):
    """Probabilités du trajet : moyenne de celles de ses fenêtres (comme à l'évaluation)."""
    return np.asarray(window_probs).mean(axis=0)
