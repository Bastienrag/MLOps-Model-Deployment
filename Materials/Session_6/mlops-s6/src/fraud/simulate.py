"""Replay "production" traffic (the current split) against the API, with optional drift.

    python -m fraud.simulate --n 500                      # normal traffic
    python -m fraud.simulate --n 500 --drift 0.8          # data drift (S6/S7)
    python -m fraud.simulate --n 500 --fraud-shift 0.7    # concept drift: only the outcome layer sees it
    python -m fraud.simulate --n 500 --amount-scale 100   # upstream unit bug: amounts sent in cents (S6)

Frauds are only 0.17 % of real traffic (about 1 in 500 requests): the outcome layer
would stay empty. Traffic is therefore resampled to --fraud-share frauds (2 % by default).
"""

from __future__ import annotations

import argparse
import random

import httpx
import pandas as pd

from fraud.config import FEATURES, ROOT, TARGET, load_params
from fraud.data import apply_concept_drift, apply_data_drift


def sample_traffic(df: pd.DataFrame, n: int, fraud_share: float, seed: int) -> pd.DataFrame:
    n_fraud = round(n * fraud_share)
    frauds = df[df[TARGET] == 1].sample(n_fraud, replace=True, random_state=seed)
    legit = df[df[TARGET] == 0].sample(n - n_fraud, random_state=seed)
    return pd.concat([frauds, legit]).sample(frac=1, random_state=seed).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--drift", type=float, default=0.0)
    parser.add_argument("--fraud-shift", type=float, default=0.0, help="share of frauds disguised as legit (0-1)")
    parser.add_argument("--fraud-share", type=float, default=0.02)
    parser.add_argument(
        "--amount-scale",
        type=float,
        default=1.0,
        help="multiply amounts before sending: 100 simulates an upstream unit bug (cents)",
    )
    parser.add_argument("--feedback-rate", type=float, default=0.3, help="share of transactions whose label comes back")
    parser.add_argument("--seed", type=int, default=random.randint(0, 10**6))
    args = parser.parse_args()

    current = pd.read_csv(ROOT / load_params()["data"]["current_path"])
    df = sample_traffic(current, args.n, args.fraud_share, args.seed)
    df = apply_concept_drift(apply_data_drift(df, args.drift, args.seed), args.fraud_shift, args.seed)
    df["amount"] = df["amount"] * args.amount_scale

    rng = random.Random(args.seed)
    errors = 0
    with httpx.Client(base_url=args.url, timeout=5) as client:
        for i, row in df.iterrows():
            payload = {"transaction_id": f"sim-{args.seed}-{i}", "customer_id": str(rng.randint(10_000, 99_999))}
            payload |= {k: float(row[k]) for k in FEATURES}
            r = client.post("/predict", json=payload)
            if r.status_code != 200:
                errors += 1
                continue
            if rng.random() < args.feedback_rate:
                client.post(
                    "/feedback",
                    json={
                        "transaction_id": payload["transaction_id"],
                        "predicted_fraud": r.json()["is_fraud"],
                        "actual_fraud": bool(row[TARGET]),
                    },
                )
    print(f"{args.n} requests sent to {args.url}, {errors} errors")


if __name__ == "__main__":
    main()
