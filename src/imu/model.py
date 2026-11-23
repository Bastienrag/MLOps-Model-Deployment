"""CNN 1D : fenêtre (temps, 6 capteurs) -> logits des 5 classes."""

import torch
from torch import nn


class IMUNet(nn.Module):
    def __init__(self, n_sensors=6, n_classes=5, channels=(32, 64, 64), dropout=0.3):
        super().__init__()
        layers, c_in = [], n_sensors
        for i, c_out in enumerate(channels):
            layers += [
                nn.Conv1d(c_in, c_out, kernel_size=7 if i == 0 else 5 if i == 1 else 3, padding="same"),
                nn.BatchNorm1d(c_out),
                nn.ReLU(),
            ]
            if i < len(channels) - 1:
                layers.append(nn.MaxPool1d(2))
            c_in = c_out
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c_in, n_classes))

    def forward(self, x):  # x : (batch, temps, capteurs)
        h = self.features(x.transpose(1, 2))
        h = torch.cat([h.mean(dim=2), h.amax(dim=2)], dim=1)  # moyenne et max dans le temps
        return self.head(h)
