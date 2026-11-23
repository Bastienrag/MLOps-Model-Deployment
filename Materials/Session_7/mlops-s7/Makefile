.DEFAULT_GOAL := help
CLUSTER := mlops
VERSION ?= v1

help: ## Command list
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- Services (S1+) ---------------------------------------------------------
setup: ## Installs the Python environment (uv sync), once per clone
	uv sync

monitoring: ## Starts Prometheus and Grafana only (S6): no S3, no MLflow needed
	docker compose up -d prometheus grafana

up: ## Starts S3, MLflow, Prometheus, Grafana
	docker compose up -d --build

down: ## Stops services (volumes are kept)
	docker compose down

# --- Model lifecycle (S1-S3) ------------------------------------------
data: ## Generates data (data/raw + data/golden)
	uv run python -m fraud.data

train: ## Trains and saves a version to MLflow
	uv run python -m fraud.train

evaluate: ## Gate AUC  on golden dataset, promote to @staging if OK
	uv run python -m fraud.evaluate

pipeline: ## Pipeline DVC (generate -> train -> evaluate)
	dvc repro

export: ## Downloads model @staging in ./model
	uv run python -m fraud.export_model

# --- Code quality ---------------------------------------------------------------
lint: ## Ruff
	uv run ruff check --fix src tests && uv run ruff format --check src tests

test: ## Tests
	uv run pytest -q

# --- Serving (S4-S5) ---------------------------------------------------------
api: ## Local API (model ./model), logs in logs/predictions.jsonl
	mkdir -p logs
	PREDICTION_LOG_PATH=logs/predictions.jsonl MODEL_URI=model uv run uvicorn fraud.api.main:app --reload --host 0.0.0.0 --port 8000

image: export ## Builds fraud-api:$(VERSION) image with model @staging
	docker build -t fraud-api:$(VERSION) .

load: ## Load test 1 min, 1000 users (results in reports/load_*)
	mkdir -p reports
	uv run locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 1000 -r 100 -t 1m --csv reports/load

# --- Kubernetes (S5, S8) -----------------------------------------------------
kind-up: ## Creates kind cluster
	kind create cluster --name $(CLUSTER) --config k8s/kind-config.yaml

kind-deploy: ## Loads fraud-api:$(VERSION) within kind and deploys v1 + service
	kind load docker-image fraud-api:$(VERSION) --name $(CLUSTER)
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
	kubectl rollout status deploy/fraud-api-v1

kind-down: ## Drops cluster
	kind delete cluster --name $(CLUSTER)

# --- Monitoring and drift (S6-S7) ---------------------------------------------
reference: ## Reference window for the drift tests (healthy production traffic)
	uv run python -m fraud.reference

simulate: ## Simulate traffic to API
	uv run python -m fraud.simulate --n 500

simulate-drift: ## Traffic with data drift
	uv run python -m fraud.simulate --n 500 --drift 0.8

simulate-unit-bug: ## Traffic with an upstream unit bug (amounts in cents)
	uv run python -m fraud.simulate --n 500 --amount-scale 100

drift: ## Drift tests on API logs + Evidently report
	uv run python -m fraud.drift --current logs/predictions.jsonl --evidently reports/drift.html

.PHONY: help setup monitoring reference up down data train evaluate pipeline export lint test api image load kind-up kind-deploy kind-down simulate simulate-drift simulate-unit-bug drift
