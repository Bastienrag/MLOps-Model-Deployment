import numpy as np
import pandas as pd
import pytest

from imu.data import SENSORS, clean_ride, split_rides, windows


def make_ride(n=10):
    df = pd.DataFrame(np.arange(n * 6, dtype=float).reshape(n, 6), columns=SENSORS)
    df.insert(0, "time_index", np.arange(n) * 100)
    return df


def test_clean_ride_interpolates_all_nan_rows():
    ride = make_ride()
    ride.loc[3:4, SENSORS] = np.nan
    out = clean_ride(ride)
    assert not np.isnan(out).any()
    assert out.shape == (10, 6)
    # interpolation linéaire entre les lignes 2 et 5
    before, after = ride.loc[2, "ax"], ride.loc[5, "ax"]
    assert out[3, 0] == pytest.approx(before + (after - before) / 3)


def test_clean_ride_fills_edges():
    ride = make_ride()
    ride.loc[0, SENSORS] = np.nan
    ride.loc[9, SENSORS] = np.nan
    out = clean_ride(ride)
    assert not np.isnan(out).any()
    assert (out[0] == out[1]).all() and (out[9] == out[8]).all()


def test_clean_ride_drops_duplicate_time_index_keeping_first():
    ride = make_ride()
    ride.loc[5, "time_index"] = ride.loc[4, "time_index"]
    out = clean_ride(ride)
    assert len(out) == 9
    assert (out[4] == ride.loc[4, SENSORS].to_numpy()).all()


def test_windows_shape_and_overlap():
    series = np.arange(1000 * 6, dtype=np.float32).reshape(1000, 6)
    w = windows(series, size=256, stride=128)
    assert w.shape == (6, 256, 6)  # (1000 - 256) // 128 + 1
    assert (w[1][:128] == w[0][128:]).all()  # 50 % de recouvrement


def test_windows_rejects_short_ride():
    with pytest.raises(ValueError):
        windows(np.zeros((100, 6), dtype=np.float32), size=256, stride=128)


def test_split_rides_is_disjoint_stratified_and_deterministic():
    classes = ["a"] * 40 + ["b"] * 10 + ["c"] * 6
    labels = pd.Series(classes, index=[f"r{i}" for i in range(len(classes))])
    split = split_rides(labels, 0.70, 0.15, seed=42)
    train, val, test = (set(split[k]) for k in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)
    assert train | val | test == set(labels.index)
    for part in (split["train"], split["val"], split["test"]):
        assert set(labels[part]) == {"a", "b", "c"}  # chaque classe est dans chaque jeu
    assert split == split_rides(labels, 0.70, 0.15, seed=42)
