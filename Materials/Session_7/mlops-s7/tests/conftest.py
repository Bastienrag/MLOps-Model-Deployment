"""Test fixtures.

Tests never download the Kaggle dataset (no credentials in CI, 150 MB): they use a small
fake file with the SAME schema (Time, V1..V28, Amount, Class) and an easy fraud signal.
"""

import mlflow
import numpy as np
import pandas as pd
import pytest

from fraud.config import TARGET
from fraud.data import load_raw
from fraud.features import SKOPS_TRUSTED_TYPES, build_pipeline, select_features

TRAIN_PARAMS = {
    "model": "hist_gradient_boosting",
    "learning_rate": 0.1,
    "max_iter": 50,
    "max_depth": 4,
    "random_state": 0,
}


def make_fake_creditcard(n: int, seed: int, fraud_rate: float = 0.02) -> pd.DataFrame:
    """Kaggle-shaped data: frauds have lower V12/V14 and higher V4, as in the real dataset."""
    rng = np.random.default_rng(seed)
    is_fraud = rng.random(n) < fraud_rate
    df = pd.DataFrame({"Time": np.sort(rng.uniform(0, 172_800, n))})
    for i in range(1, 29):
        df[f"V{i}"] = rng.normal(0, 1, n)
    df.loc[is_fraud, "V14"] -= 4
    df.loc[is_fraud, "V12"] -= 3
    df.loc[is_fraud, "V4"] += 2
    df["Amount"] = rng.lognormal(3.1, 1.3, n).round(2)
    df["Class"] = is_fraud.astype(int)
    return df


@pytest.fixture(scope="session")
def raw_csv(tmp_path_factory):
    path = tmp_path_factory.mktemp("raw") / "creditcard.csv"
    make_fake_creditcard(6000, seed=1).to_csv(path, index=False)
    return path


@pytest.fixture(scope="session")
def small_data(raw_csv) -> pd.DataFrame:
    return load_raw(raw_csv)


@pytest.fixture(scope="session")
def trained_pipeline(small_data):
    return build_pipeline(TRAIN_PARAMS).fit(select_features(small_data), small_data[TARGET])


@pytest.fixture(scope="session")
def model_dir(tmp_path_factory, trained_pipeline):
    path = tmp_path_factory.mktemp("model") / "model"
    mlflow.sklearn.save_model(trained_pipeline, str(path), skops_trusted_types=SKOPS_TRUSTED_TYPES)
    (path / "model_meta.json").write_text('{"version": "test", "decision_threshold": 0.3}')
    return path
