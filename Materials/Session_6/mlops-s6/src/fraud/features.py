"""Preprocessing + model in ONE sklearn pipeline.

Anti-pattern avoided: preprocessing separately in a notebook, then serving a model
that expects already-transformed columns. Here the MLflow artifact holds everything
and the API passes raw fields.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fraud.config import FEATURES, PCA_FEATURES

# MLflow 3 serialises sklearn models with skops: only types declared as trusted are
# loaded back (unlike pickle, which can execute arbitrary code).
SKOPS_TRUSTED_TYPES = ["sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]


def build_pipeline(params: dict) -> Pipeline:
    # V1..V28 are already PCA components; only `amount` is on a raw scale.
    preprocess = ColumnTransformer(
        [
            ("amount", StandardScaler(), ["amount"]),
            ("pca", "passthrough", PCA_FEATURES),
        ]
    )
    if params["model"] == "logistic_regression":
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=params["learning_rate"],
            max_iter=params["max_iter"],
            max_depth=params["max_depth"],
            random_state=params["random_state"],
        )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep model columns only (transaction_id, customer_id and time never enter the model)."""
    return df[FEATURES]


def pick_threshold(y_true, proba, min_recall: float) -> float:
    """Highest threshold that still keeps recall >= min_recall (i.e. best precision at that recall)."""
    _, recall, thresholds = precision_recall_curve(y_true, proba)
    ok = np.where(recall[:-1] >= min_recall)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])
