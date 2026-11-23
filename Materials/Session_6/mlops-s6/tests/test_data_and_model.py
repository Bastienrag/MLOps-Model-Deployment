from conftest import make_fake_creditcard
from sklearn.metrics import roc_auc_score

from fraud.config import FEATURES, TARGET
from fraud.data import chronological_split, load_raw
from fraud.features import pick_threshold, select_features


def test_load_raw_normalises_kaggle_columns(small_data):
    assert set(FEATURES) | {TARGET, "time", "transaction_id"} <= set(small_data.columns)
    assert not any(c[0].isupper() for c in small_data.columns)


def test_split_is_chronological_and_complete(small_data):
    parts = chronological_split(small_data, 0.7, 0.15)
    assert sum(len(p) for p in parts.values()) == len(small_data)
    assert parts["train"]["time"].max() <= parts["golden"]["time"].min()
    assert parts["golden"]["time"].max() <= parts["current"]["time"].min()


def test_model_beats_gate_on_fresh_data(trained_pipeline, tmp_path):
    path = tmp_path / "holdout.csv"
    make_fake_creditcard(4000, seed=99).to_csv(path, index=False)
    holdout = load_raw(path)
    proba = trained_pipeline.predict_proba(select_features(holdout))[:, 1]
    assert roc_auc_score(holdout[TARGET], proba) > 0.85


def test_threshold_keeps_requested_recall():
    y = [0, 0, 0, 0, 1, 1, 1, 1]
    proba = [0.1, 0.2, 0.3, 0.6, 0.4, 0.7, 0.8, 0.9]
    t = pick_threshold(y, proba, min_recall=0.75)
    recall = sum(p >= t for p, label in zip(proba, y, strict=True) if label) / 4
    assert recall >= 0.75 and t == 0.7
