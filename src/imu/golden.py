"""Jeu figé pour le gate de la CI : les trajets de test, une fenêtre sur deux, en float16.

    python -m imu.golden

Les runners GitHub n'ont accès ni au remote DVC ni au registry (ils sont locaux). Ce petit fichier est
donc versionné dans Git avec le modèle : la CI évalue le vrai modèle sur de vraies données, sans rien
télécharger. Une fenêtre sur deux (pas de 256 au lieu de 128) divise la taille par deux sans perdre un
trajet.
"""

from pathlib import Path

import numpy as np
import yaml

from imu.baseline import load_split


def main():
    cfg = yaml.safe_load(Path("params.yaml").read_text())
    X, y, ride_id = load_split(cfg["data"]["processed_dir"], "test")
    keep = np.zeros(len(X), dtype=bool)
    for ride in np.unique(ride_id):
        idx = np.flatnonzero(ride_id == ride)
        keep[idx[:: cfg["golden"]["every"]]] = True
    out = Path(cfg["golden"]["path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, X=X[keep].astype(np.float16), y=y[keep], ride_id=ride_id[keep])
    size = out.stat().st_size / 1e6
    print(f"{keep.sum()} fenêtres de {len(np.unique(ride_id))} trajets -> {out} ({size:.1f} Mo)")


if __name__ == "__main__":
    main()
