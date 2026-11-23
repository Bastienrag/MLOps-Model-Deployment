#!/usr/bin/env bash
# Cluster Kubernetes local (kind). Usage :
#   k8s/deploy.sh up        crée le cluster
#   k8s/deploy.sh deploy    construit imu-api:<commit>, le charge dans kind et déploie
#   k8s/deploy.sh status    pods, endpoints du Service et santé
#   k8s/deploy.sh down      supprime le cluster
set -euo pipefail
cd "$(dirname "$0")/.."
export PATH="$HOME/.local/bin:$PATH"
CLUSTER="${CLUSTER:-imu}"
VERSION="${VERSION:-$(git rev-parse --short HEAD)}"

case "${1:-}" in
  up)
    kind create cluster --name "$CLUSTER" --config k8s/kind-config.yaml
    ;;
  deploy)
    docker build --build-arg GIT_SHA="$(git rev-parse HEAD)" -t "imu-api:$VERSION" .
    kind load docker-image "imu-api:$VERSION" --name "$CLUSTER"
    kubectl apply -f k8s/configmap.yaml -f k8s/deployment.yaml -f k8s/service.yaml
    kubectl set image deployment/imu-api "api=imu-api:$VERSION"
    kubectl rollout status deployment/imu-api --timeout=180s
    ;;
  status)
    kubectl get pods -o wide
    kubectl get endpoints imu-api
    curl -s http://localhost:8080/health; echo
    ;;
  down)
    kind delete cluster --name "$CLUSTER"
    ;;
  *)
    sed -n '2,7p' "$0"
    exit 1
    ;;
esac
