import numpy as np
import pytest

from imu.inference import aggregate, preprocess


def ride(n=600, seed=0):
    return np.random.default_rng(seed).normal(size=(n, 6))


def test_preprocess_shape():
    w = preprocess(ride(600), np.zeros(6), np.ones(6), window=256, stride=128)
    assert w.shape == (3, 256, 6)
    assert w.dtype == np.float32


def test_preprocess_normalizes_with_given_statistics():
    values = ride()
    mean, std = np.arange(6, dtype=np.float32), np.full(6, 2.0, dtype=np.float32)
    raw = preprocess(values, np.zeros(6), np.ones(6), 256, 128)
    scaled = preprocess(values, mean, std, 256, 128)
    assert scaled == pytest.approx((raw - mean) / std, abs=1e-5)


def test_preprocess_fills_missing_rows():
    values = ride()
    values[10:13] = np.nan
    w = preprocess(values, np.zeros(6), np.ones(6), 256, 128)
    assert not np.isnan(w).any()


def test_aggregate_averages_window_probabilities():
    probs = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    assert aggregate(probs).tolist() == [0.5, 0.5]
