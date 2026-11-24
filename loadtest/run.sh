#!/usr/bin/env bash
# Même profil de charge pour 1 puis 3 replicas : paliers d'utilisateurs de 45 s chacun.
#   loadtest/run.sh [hôte]   -> loadtest/results/replicas-<n>/users-<u>_stats.csv
set -euo pipefail
cd "$(dirname "$0")/.."
host="${1:-http://localhost:8080}"
for replicas in 1 3; do
  kubectl scale deployment/imu-api --replicas="$replicas"
  kubectl rollout status deployment/imu-api --timeout=120s
  sleep 10
  for users in 5 10 20 40 80; do
    out="loadtest/results/replicas-$replicas"
    mkdir -p "$out"
    uv run --group load locust -f loadtest/locustfile.py --headless -u "$users" -r 5 -t 45s \
      --host "$host" --csv "$out/users-$users" --only-summary > "$out/users-$users.log" 2>&1
    echo "replicas=$replicas users=$users fait"
  done
done
kubectl scale deployment/imu-api --replicas=3
