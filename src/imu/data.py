"""Prétraitement des trajets IMU : nettoyage, découpage par trajet, fenêtrage.

    python -m imu.data

Entrées : GBG500.parquet (une ligne par mesure) et GBG500_labels.csv (un label par trajet).
Sorties dans data/processed/ : train.npz, val.npz, test.npz, splits.csv, meta.json.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

SENSORS = ["ax", "ay", "az", "rx", "ry", "rz"]


def load(raw_path, labels_path):
    """Lit les mesures et les labels, et vérifie qu'ils décrivent les mêmes trajets."""
    raw = pd.read_parquet(raw_path)
    labels = pd.read_csv(labels_path).set_index("RIDE_ID")["LABEL"]
    if labels.index.has_duplicates:
        raise ValueError("RIDE_ID en double dans les labels")
    if set(raw["ride_id"]) != set(labels.index):
        raise ValueError("Les trajets des mesures et des labels ne correspondent pas")
    return raw, labels


def clean_ride(ride):
    """Renvoie le trajet sous forme d'un tableau (temps, capteurs) sans trou.

    - tri par time_index ;
    - un time_index en double (deux mesures au même instant) : on garde la première ;
    - lignes dont les 6 capteurs sont NaN : interpolation linéaire dans le temps,
      puis recopie de la valeur voisine si le trou est au début ou à la fin.
    """
    ride = ride.sort_values("time_index", kind="stable")
    ride = ride.drop_duplicates("time_index", keep="first")
    values = ride[SENSORS].interpolate(method="linear", limit_direction="both")
    return values.to_numpy(dtype=np.float32)


def split_rides(labels, train_share, val_share, seed):
    """Découpe les trajets (jamais les fenêtres) en train/val/test, stratifié par classe."""
    rides = labels.index.to_numpy()
    train, rest = train_test_split(
        rides, train_size=train_share, stratify=labels.loc[rides], random_state=seed
    )
    test_share = 1 - train_share - val_share
    val, test = train_test_split(
        rest,
        test_size=test_share / (val_share + test_share),
        stratify=labels.loc[rest],
        random_state=seed,
    )
    return {"train": list(train), "val": list(val), "test": list(test)}


def windows(series, size, stride):
    """Découpe un trajet (temps, capteurs) en fenêtres (n, size, capteurs).

    Le reste de fin de trajet, plus court qu'une fenêtre, est abandonné.
    """
    if len(series) < size:
        raise ValueError(f"trajet de {len(series)} points, plus court que la fenêtre ({size})")
    starts = range(0, len(series) - size + 1, stride)
    return np.stack([series[s : s + size] for s in starts])


def fit_stats(rides):
    """Moyenne et écart-type par capteur, sur les mesures des trajets d'entraînement."""
    flat = np.concatenate(rides)
    return flat.mean(axis=0), flat.std(axis=0)


def build_split(ride_ids, series, labels, classes, mean, std, size, stride):
    """Fenêtres normalisées d'un jeu : X, y (un label par fenêtre) et ride_id de chaque fenêtre."""
    xs, ys, owners = [], [], []
    for ride_id in ride_ids:
        w = (windows(series[ride_id], size, stride) - mean) / std
        xs.append(w.astype(np.float32))
        ys.append(np.full(len(w), classes.index(labels[ride_id]), dtype=np.int8))
        owners.append(np.full(len(w), ride_id))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(owners)


def prepare(params):
    raw, labels = load(params["raw_path"], params["labels_path"])
    classes = params["classes"]
    unknown = set(labels) - set(classes)
    if unknown:
        raise ValueError(f"Labels hors de la liste des classes : {unknown}")

    series = {ride_id: clean_ride(g) for ride_id, g in raw.groupby("ride_id")}
    split = split_rides(labels, params["train_share"], params["val_share"], params["seed"])

    # Les statistiques de normalisation viennent du train uniquement (pas de fuite).
    mean, std = fit_stats([series[r] for r in split["train"]])

    out = Path(params["processed_dir"])
    out.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, ride_ids in split.items():
        X, y, owners = build_split(
            ride_ids, series, labels, classes, mean, std, params["window"], params["stride"]
        )
        np.savez_compressed(out / f"{name}.npz", X=X, y=y, ride_id=owners)
        counts[name] = {
            "rides": len(ride_ids),
            "windows": int(len(X)),
            "rides_per_class": {c: int((labels[ride_ids] == c).sum()) for c in classes},
        }

    rows = [(r, labels[r], name) for name, ids in split.items() for r in ids]
    pd.DataFrame(rows, columns=["ride_id", "label", "split"]).sort_values("ride_id").to_csv(
        out / "splits.csv", index=False
    )
    meta = {
        "classes": classes,
        "sensors": SENSORS,
        "window": params["window"],
        "stride": params["stride"],
        "mean": mean.tolist(),
        "std": std.tolist(),
        "counts": counts,
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")
    return meta


if __name__ == "__main__":
    prepared = prepare(yaml.safe_load(Path("params.yaml").read_text())["data"])
    for split_name, c in prepared["counts"].items():
        print(f"{split_name:5s} {c['rides']:4d} trajets {c['windows']:6d} fenêtres")
        print(f"      {c['rides_per_class']}")
