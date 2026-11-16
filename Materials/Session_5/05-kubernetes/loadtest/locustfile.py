"""Load test (S5).

    locust -f loadtest/locustfile.py --host http://localhost:8000 \
           --headless -u 50 -r 10 -t 1m --csv reports/load

Then read reports/load_stats.csv: columns "95%" (P95, ms) and "Requests/s".
Payloads are real transactions from the golden split (run `make pipeline` first).
"""

import random

import pandas as pd
from locust import HttpUser, between, task

from fraud.config import FEATURES, ROOT, load_params

_golden = pd.read_csv(ROOT / load_params()["data"]["golden_path"], usecols=FEATURES, nrows=5000)
POOL = [{"transaction_id": f"load-{i}", **row} for i, row in enumerate(_golden.to_dict(orient="records"))]


class Scorer(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(20)
    def predict(self):
        self.client.post("/predict", json=random.choice(POOL))

    @task(1)
    def health(self):
        self.client.get("/health")
