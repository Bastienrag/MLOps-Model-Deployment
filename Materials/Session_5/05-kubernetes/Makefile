.DEFAULT_GOAL := help
CLUSTER := mlops
VERSION ?= v1
CSV ?= reports/load_k8s

help: ## Command list
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

api: ## Local API on port 8000 (model in ./model), logs in logs/predictions.jsonl
	PREDICTION_LOG_PATH=logs/predictions.jsonl MODEL_URI=model uv run uvicorn fraud.api.main:app --host 0.0.0.0 --port 8000

image: ## Builds fraud-api:$(VERSION) (the model is already in ./model)
	docker build -t fraud-api:$(VERSION) .

load: ## Load test on the local API (port 8000): 1 min, 50 users
	mkdir -p reports && uv run locust -f loadtest/locustfile.py --host http://localhost:8000 --headless -u 50 -r 10 -t 1m --csv reports/load_local

load-kind: ## Load test on the kind cluster (port 8080). Example: make load-kind CSV=reports/load_k8s_1r
	mkdir -p reports && uv run locust -f loadtest/locustfile.py --host http://localhost:8080 --headless -u 50 -r 10 -t 1m --csv $(CSV)

kind-up: ## Creates the kind cluster
	kind create cluster --name $(CLUSTER) --config k8s/kind-config.yaml

kind-deploy: ## Loads fraud-api:$(VERSION) into kind and deploys v1 + service
	kind load docker-image fraud-api:$(VERSION) --name $(CLUSTER)
	kubectl apply -f k8s/deployment.yaml -f k8s/service.yaml
	kubectl rollout status deploy/fraud-api-v1

kind-down: ## Drops the cluster
	kind delete cluster --name $(CLUSTER)
