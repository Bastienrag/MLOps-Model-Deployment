"""Reference window for the drift tests: a healthy stretch of production traffic (S7).

    python -m fraud.reference        # writes data/processed/reference.csv (deterministic)

The drift tests compare the API logs with this window. The training split is NOT the reference:
the split is chronological, so train.csv and the production period come from different moments
of the dataset. Measured against train.csv, even normal traffic shows strong drift on V1, V3 and V28
(PSI around 0.9). That gap is real, but it is a gap between two periods, not a change in the traffic
the model serves now: it cannot be the baseline of an alert.
"""

from __future__ import annotations

import pandas as pd

from fraud.config import ROOT, load_params

SIZE = 20_000
SEED = 0


def main() -> None:
    params = load_params()
    current = pd.read_csv(ROOT / params["data"]["current_path"])
    reference = current.sample(SIZE, random_state=SEED).sort_index()
    out = ROOT / params["drift"]["reference_path"]
    out.parent.mkdir(parents=True, exist_ok=True)
    reference.to_csv(out, index=False)
    print(f"reference: {len(reference)} rows -> {out}")


if __name__ == "__main__":
    main()
