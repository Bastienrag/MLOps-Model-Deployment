"""Extrait un court trajet d'exemple par classe (jeu de test) pour la web app.

    python -m imu.web.make_examples

Chaque exemple garde les 600 premières mesures brutes (lignes manquantes comprises, en null) : c'est
ce qu'un client enverrait à l'API. Les identifiants réels des trajets ne sont pas conservés.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from imu.data import SENSORS, load

N_POINTS = 600
OUT = Path(__file__).parent / "static" / "examples"


def main():
    params = yaml.safe_load(Path("params.yaml").read_text())["data"]
    raw, labels = load(params["raw_path"], params["labels_path"])
    splits = pd.read_csv(Path(params["processed_dir"]) / "splits.csv")
    test = splits[splits["split"] == "test"].sort_values("ride_id")

    OUT.mkdir(exist_ok=True)
    manifest = []
    for cls in params["classes"]:
        ride_id = test[test["label"] == cls]["ride_id"].iloc[0]
        ride = raw[raw["ride_id"] == ride_id].sort_values("time_index").head(N_POINTS)
        values = ride[SENSORS].round(3).astype(object).where(ride[SENSORS].notna(), None)
        slug = cls.lower().replace(" ", "-")
        example = {
            "ride_id": f"exemple-{slug}",
            "true_class": cls,
            "columns": SENSORS,
            "rows": values.to_numpy().tolist(),
        }
        (OUT / f"{slug}.json").write_text(json.dumps(example, separators=(",", ":"), allow_nan=False))
        manifest.append({"file": f"{slug}.json", "true_class": cls, "label": f"{cls} (réel)"})
        assert labels[ride_id] == cls and not np.isnan(ride[SENSORS].to_numpy()).all()
    (OUT / "index.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(f"{len(manifest)} exemples dans {OUT}")


if __name__ == "__main__":
    main()
