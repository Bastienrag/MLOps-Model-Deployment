"""Drift detection: Kolmogorov-Smirnov (continuous), chi-squared (categorical), PSI (binned).

    python -m fraud.drift --current logs/predictions.jsonl
    python -m fraud.drift --current logs/predictions.jsonl --evidently reports/drift.html

Exit code 3 when a PSI exceeds the retraining threshold (params.yaml). Chain on that code only,
never with ||: any other failure would also start a training run (see docs/sessions/s7.md, step 4).

Keep in mind: these tests compare INPUT distributions. They see data drift, not concept
drift (a change in the input -> fraud relationship), which only shows up with labels
(the "outcome" monitoring layer).

The Kaggle dataset has no categorical column (V1..V28 are PCA components), so the
chi-squared test is shown on amount bands. In a real system it would apply to
categories such as country or merchant type.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from scipy import stats

from fraud.config import FEATURES, ROOT, load_params

EPS = 1e-6
AMOUNT_BANDS = [-np.inf, 5, 20, 50, 100, 250, 1000, np.inf]


def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """PSI on bins defined by the reference quantiles."""
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, edges)[0] / len(reference)
    cur = np.histogram(current, edges)[0] / len(current)
    ref, cur = np.clip(ref, EPS, None), np.clip(cur, EPS, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def chi2_pvalue(reference: pd.Series, current: pd.Series) -> float:
    table = pd.crosstab(
        np.r_[np.zeros(len(reference)), np.ones(len(current))],
        np.r_[reference.astype(str), current.astype(str)],
    )
    return float(stats.chi2_contingency(table)[1])


def compare(reference: pd.DataFrame, current: pd.DataFrame, params: dict | None = None) -> dict:
    params = params or load_params()["drift"]
    features = {}
    for col in FEATURES:
        ks = stats.ks_2samp(reference[col], current[col])
        features[col] = {
            "test": "ks",
            "statistic": float(ks.statistic),
            "pvalue": float(ks.pvalue),
            "psi": psi(reference[col], current[col]),
        }
    bands = [pd.cut(df["amount"], AMOUNT_BANDS) for df in (reference, current)]
    features["amount_band"] = {"test": "chi2", "pvalue": chi2_pvalue(*bands), "psi": None}

    for f in features.values():
        f["drift_detected"] = f["pvalue"] < params["pvalue"]
        if f["psi"] is None:
            f["level"] = "n/a"
        elif f["psi"] > params["psi_retrain"]:
            f["level"] = "retrain"
        else:
            f["level"] = "alert" if f["psi"] > params["psi_alert"] else "ok"
    max_psi = max(f["psi"] for f in features.values() if f["psi"] is not None)
    return {
        "n_reference": len(reference),
        "n_current": len(current),
        "max_psi": max_psi,
        "retrain": max_psi > params["psi_retrain"],
        "features": features,
    }


def load_current(path: Path) -> pd.DataFrame:
    if path.suffix == ".jsonl":  # API logs: keep prediction events only
        events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return pd.DataFrame([e for e in events if e.get("event") == "prediction"])
    return pd.read_csv(path)


def evidently_report(reference: pd.DataFrame, current: pd.DataFrame, out: Path) -> None:
    snapshot = Report([DataDriftPreset()]).run(reference_data=reference[FEATURES], current_data=current[FEATURES])
    out.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(out))


def main() -> int:
    params = load_params()
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=ROOT / params["drift"]["reference_path"])
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--evidently", type=Path, default=None, help="path of the Evidently HTML report")
    args = parser.parse_args()

    reference, current = pd.read_csv(args.reference), load_current(args.current)
    result = compare(reference, current, params["drift"])

    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "drift.json").write_text(json.dumps(result, indent=2))
    for name, f in sorted(result["features"].items(), key=lambda kv: -(kv[1]["psi"] or 0)):
        psi_txt = f"PSI={f['psi']:.3f}" if f["psi"] is not None else "PSI=  n/a"
        print(f"{name:12s} {f['test']:5s} p={f['pvalue']:.3g}  {psi_txt}  {f['level']}")
    print(f"max PSI = {result['max_psi']:.3f} -> {'RETRAIN' if result['retrain'] else 'ok'}")

    if args.evidently:
        evidently_report(reference, current, args.evidently)
        print(f"Evidently report -> {args.evidently}")
    return 3 if result["retrain"] else 0


if __name__ == "__main__":
    sys.exit(main())
