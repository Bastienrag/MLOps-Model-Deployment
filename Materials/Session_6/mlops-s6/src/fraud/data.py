"""Data ingestion for the Kaggle "Credit Card Fraud Detection" dataset (mlg-ulb/creditcardfraud).

    python -m fraud.data download   # Kaggle -> data/raw/creditcard.csv (needs a Kaggle API token)
    python -m fraud.data split      # data/raw -> data/processed/{train,golden,current}.csv

The split is CHRONOLOGICAL (on `time`), never random: the model is trained on the
past and evaluated on the future, as it will be in production.

    train    first 70 %   fit + threshold selection
    golden   next 15 %    frozen evaluation set used by the CI gate
    current  last 15 %    "production" traffic replayed by fraud.simulate
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd

from fraud.config import KAGGLE_DATASET, PCA_FEATURES, ROOT, TARGET, load_params


def download(out: Path) -> Path:
    """Download creditcard.csv from Kaggle. Credentials: KAGGLE_API_TOKEN, or ~/.kaggle/."""
    if out.exists():
        print(f"{out} already exists, skipping download")
        return out
    folder = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(folder / "creditcard.csv", out)
    print(f"{KAGGLE_DATASET} -> {out}")
    return out


def load_raw(path: Path) -> pd.DataFrame:
    """Read the Kaggle CSV and normalise column names (the only place they are renamed)."""
    df = pd.read_csv(path)
    df = df.rename(columns={"Time": "time", "Amount": "amount", "Class": TARGET})
    df = df.rename(columns={f"V{i}": f"v{i}" for i in range(1, 29)})
    df.insert(0, "transaction_id", [f"tx-{i:06d}" for i in range(len(df))])
    return df


def chronological_split(df: pd.DataFrame, train: float, golden: float) -> dict[str, pd.DataFrame]:
    df = df.sort_values("time", kind="stable").reset_index(drop=True)
    n_train, n_golden = int(len(df) * train), int(len(df) * golden)
    return {
        "train": df.iloc[:n_train],
        "golden": df.iloc[n_train : n_train + n_golden],
        "current": df.iloc[n_train + n_golden :],
    }


# --- Simulated shifts, used by fraud.simulate and the tests ----------------------


def apply_data_drift(df: pd.DataFrame, drift: float, seed: int = 0) -> pd.DataFrame:
    """Data drift: inputs move (higher amounts, shifted components), the fraud mechanism does not."""
    if drift == 0:
        return df
    rng = np.random.default_rng(seed)
    out = df.copy()
    out["amount"] = (out["amount"] * (1 + 1.5 * drift)).round(2)
    for col in ["v1", "v2", "v3", "v4"]:
        out[col] = out[col] + drift * out[col].std() + rng.normal(0, 0.1, len(out))
    return out


def apply_concept_drift(df: pd.DataFrame, share: float, seed: int = 0) -> pd.DataFrame:
    """Concept drift: a share of fraudsters now mimic legitimate behaviour.

    Their components are replaced by those of random legitimate transactions while
    the label stays 1. Input distributions barely move (frauds are rare): only the
    outcome layer (recall on delayed labels) can see it.
    """
    if share == 0:
        return df
    rng = np.random.default_rng(seed)
    out = df.copy()
    frauds = out.index[out[TARGET] == 1]
    legit = out.index[out[TARGET] == 0]
    disguised = rng.choice(frauds, size=int(len(frauds) * share), replace=False)
    donors = rng.choice(legit, size=len(disguised), replace=True)
    out.loc[disguised, PCA_FEATURES] = out.loc[donors, PCA_FEATURES].to_numpy()
    return out


def main() -> None:
    p = load_params()["data"]
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["download", "split", "all"], nargs="?", default="all")
    args = parser.parse_args()

    raw = ROOT / p["raw_path"]
    if args.step in ("download", "all"):
        download(raw)
    if args.step in ("split", "all"):
        parts = chronological_split(load_raw(raw), p["train_share"], p["golden_share"])
        for name, part in parts.items():
            path = ROOT / p[f"{name}_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            part.to_csv(path, index=False)
            print(f"{name:8s} {len(part):7d} rows, fraud rate {part[TARGET].mean():.3%} -> {path}")


if __name__ == "__main__":
    main()
