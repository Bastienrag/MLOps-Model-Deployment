"""Prometheus metrics, one family per monitoring layer."""

from prometheus_client import Counter, Gauge, Histogram

# Layer 1: infrastructure
REQUESTS = Counter("api_requests_total", "HTTP requests", ["endpoint", "status"])
LATENCY = Histogram(
    "api_request_duration_seconds",
    "Request latency",
    ["endpoint"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2),
)

# Layer 2: input distribution
INPUT_AMOUNT = Histogram(
    "input_amount_euros",
    "Amount of incoming transactions",
    buckets=(1, 5, 10, 20, 50, 100, 250, 1000, 5000),
)
# A few components, among the most predictive ones on this dataset
INPUT_COMPONENT = Histogram(
    "input_component_value",
    "Value of selected PCA components",
    ["component"],
    buckets=(-20, -10, -5, -2, -1, 0, 1, 2, 5),
)
MONITORED_COMPONENTS = ("v4", "v12", "v14")

# Layer 3: prediction distribution
PREDICTION_PROBA = Histogram(
    "prediction_fraud_probability",
    "Predicted fraud probability",
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 0.8, 0.95, 1.0),
)
PREDICTED_FRAUD = Counter("prediction_fraud_total", "Transactions predicted as fraud")

# Layer 4: business outcome (labels received later)
FEEDBACK = Counter("feedback_total", "True labels received", ["outcome"])  # tp, fp, fn, tn

MODEL_INFO = Gauge("model_info", "Model version being served", ["version"])
