"""Entraînement du CNN 1D sur les fenêtres, évaluation par fenêtre et par trajet.

    python -m imu.train
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from imu.baseline import load_split
from imu.evaluate import metrics, report
from imu.model import IMUNet


def set_seed(seed, threads):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)


def class_weights(y, n_classes, power):
    """Poids ∝ (1 / effectif) ** power, normalisés pour une moyenne de 1 sur les fenêtres."""
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = np.where(counts > 0, counts, np.inf) ** -power
    w = np.where(np.isfinite(w), w, 0.0)
    return torch.tensor(w / (w[y].mean()), dtype=torch.float32)


def augment(xb, noise_std, scale_jitter):
    """Bruit gaussien et petite variation d'échelle par capteur, pour limiter le surapprentissage."""
    scale = 1 + scale_jitter * torch.randn(xb.shape[0], 1, xb.shape[2])
    return xb * scale + noise_std * torch.randn_like(xb)


@torch.no_grad()
def predict(model, X, batch_size=512):
    model.eval()
    out = [torch.softmax(model(torch.from_numpy(X[i : i + batch_size])), dim=1)
           for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy()


def fit(Xtr, ytr, tp, n_classes, on_epoch=None):
    """Entraîne un IMUNet. `on_epoch(model, epoch, loss)` est appelé après chaque époque ;
    s'il renvoie True, l'entraînement s'arrête."""
    set_seed(tp["seed"], tp["threads"])
    model = IMUNet(Xtr.shape[2], n_classes, tp["channels"], tp["dropout"])
    loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr, n_classes, tp["class_weight_power"]))
    opt = torch.optim.AdamW(model.parameters(), lr=tp["learning_rate"], weight_decay=tp["weight_decay"])
    loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr.astype(np.int64))),
        batch_size=tp["batch_size"], shuffle=True, generator=torch.Generator().manual_seed(tp["seed"]),
    )
    for epoch in range(1, tp["epochs"] + 1):
        model.train()
        total = 0.0
        for xb, yb in loader:
            xb = augment(xb, tp["noise_std"], tp["scale_jitter"])
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()
            total += loss.item() * len(xb)
        if on_epoch is not None and on_epoch(model, epoch, total / len(Xtr)):
            break
    return model


def run(params, tp):
    classes = params["classes"]
    Xtr, ytr, _ = load_split(params["processed_dir"], "train")
    Xva, yva, rva = load_split(params["processed_dir"], "val")
    Xte, yte, rte = load_split(params["processed_dir"], "test")

    state = {"best": -1.0, "state": None, "stale": 0, "history": []}

    def on_epoch(model, epoch, loss):
        # Le F1 macro par fenêtre sert à choisir l'époque : plus stable que par trajet (1715 fenêtres).
        f1 = metrics(yva, predict(model, Xva).argmax(axis=1), classes)["f1_macro"]
        state["history"].append({"epoch": epoch, "train_loss": loss, "val_f1_macro_window": f1})
        print(f"epoch {epoch:2d}  loss {loss:.4f}  val F1 macro (fenêtres) {f1:.3f}")
        if f1 > state["best"]:
            state["best"], state["stale"] = f1, 0
            state["state"] = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            state["stale"] += 1
        return state["stale"] >= tp["patience"]

    model = fit(Xtr, ytr, tp, len(classes), on_epoch)
    model.load_state_dict(state["state"])
    history = state["history"]
    results = {
        "best_val_f1_macro_window": state["best"],
        "best_epoch": max(history, key=lambda h: h["val_f1_macro_window"])["epoch"],
        "val": report(predict(model, Xva), yva, rva, classes),
        "test": report(predict(model, Xte), yte, rte, classes),
        "history": history,
    }
    return model, results


if __name__ == "__main__":
    cfg = yaml.safe_load(Path("params.yaml").read_text())
    model, results = run(cfg["data"], cfg["train"])
    Path("models").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "models/cnn.pt")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/cnn.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    for split in ("val", "test"):
        for level in ("window", "ride"):
            m = results[split][level]
            print(f"{split:4s} {level:6s} F1 macro {m['f1_macro']:.3f}  "
                  f"balanced acc {m['balanced_accuracy']:.3f}")
