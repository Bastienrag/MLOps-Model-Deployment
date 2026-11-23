"""Paquet de service du modèle : models/meta.json, à côté de models/cnn.pt.

    python -m imu.bundle

L'image Docker de l'API n'a pas MLflow : elle charge le modèle depuis ces deux fichiers. meta.json réunit
ce que le service doit connaître : classes, taille de fenêtre et pas, moyenne et écart-type de
normalisation (calculés sur le train), architecture du réseau.
"""

import json
from pathlib import Path

import yaml


def main():
    cfg = yaml.safe_load(Path("params.yaml").read_text())
    meta = json.loads((Path(cfg["data"]["processed_dir"]) / "meta.json").read_text())
    bundle = {
        key: meta[key] for key in ("classes", "sensors", "window", "stride", "mean", "std")
    } | {"channels": cfg["train"]["channels"], "dropout": cfg["train"]["dropout"]}
    out = Path("models/meta.json")
    out.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n")
    print(f"{out} : {len(bundle['classes'])} classes, fenêtre {bundle['window']}, pas {bundle['stride']}")


if __name__ == "__main__":
    main()
