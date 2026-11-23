import numpy as np
import pytest
import torch

from imu.evaluate import aggregate_by_ride, metrics
from imu.model import IMUNet
from imu.train import augment, class_weights, set_seed


def test_model_output_shape():
    model = IMUNet(n_sensors=6, n_classes=5, channels=(8, 16, 16))
    assert model(torch.zeros(4, 256, 6)).shape == (4, 5)


def test_model_accepts_other_window_lengths():
    model = IMUNet(channels=(8, 16, 16)).eval()
    assert model(torch.zeros(2, 100, 6)).shape == (2, 5)


def test_class_weights_favor_rare_classes_and_average_one():
    y = np.array([0] * 90 + [1] * 10)
    w = class_weights(y, n_classes=3, power=1.0)
    assert w[1] > w[0]
    assert w[2] == 0  # classe absente
    assert w[torch.from_numpy(y)].mean() == pytest.approx(1.0)


def test_class_weights_power_zero_is_uniform():
    w = class_weights(np.array([0, 0, 0, 1]), n_classes=2, power=0.0)
    assert w[0] == pytest.approx(w[1])


def test_same_seed_gives_same_training_step():
    def one_step():
        set_seed(0, threads=1)
        model = IMUNet(channels=(4, 4, 4))
        x = augment(torch.randn(8, 64, 6), 0.1, 0.1)
        return model(x).sum().item()

    assert one_step() == one_step()


def test_aggregate_by_ride_averages_window_probabilities():
    probs = np.array([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]])
    y = np.array([0, 0, 1])
    ride = np.array(["a", "a", "b"])
    p, ry = aggregate_by_ride(probs, y, ride)
    assert p.tolist() == [[0.5, 0.5], [0.0, 1.0]]
    assert ry.tolist() == [0, 1]


def test_metrics_balanced_accuracy_ignores_class_size():
    y_true = [0] * 8 + [1] * 2
    m = metrics(y_true, [0] * 10, ["a", "b"])  # tout prédit en "a"
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["balanced_accuracy"] == pytest.approx(0.5)
    assert m["recall_per_class"] == {"a": 1.0, "b": 0.0}
