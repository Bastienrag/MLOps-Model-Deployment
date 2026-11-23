# Runbooks

An alert without a runbook wakes someone up who does not know what to do. Each alert in
`infra/prometheus/alerts.yml` points to a section below. Each group completes them.

## Latency

Alert `LatencyP95High`: `/predict` P95 above 200 ms for 1 min.

1. Grafana, layer 1: did throughput go up at the same time? If so, it is a capacity problem.
2. Capacity. In the monitoring lab (one `make api` process), the only lever is the load: stop the traffic (`Ctrl+C` on `make load`) and check that P95 comes back under 200 ms. On Kubernetes (S5), add replicas: `kubectl scale deploy/fraud-api-v1 --replicas=<n+2>`, then check that P95 comes back down.
3. No throughput increase: look at `inference_ms` in the logs. Was a heavier model deployed? If so, [roll back](#rollback).

## Errors

Alert `ErrorRateHigh`: more than 5 % of 5xx responses.

1. `GET /health`: `degraded` means the model is not loaded. Read the startup logs (`model_load_failed`).
2. 503s right after a deployment: [roll back](#rollback) immediately, investigate afterwards.

## Drift

Alerts `InputAmountShift` and `PredictedFraudRateHigh`.

1. `make drift`: which features exceed the PSI threshold?
2. Real change (season, new market) or upstream bug (unit, shifted column)? An upstream bug is fixed at the source, not by retraining.
3. Real change with PSI > 0.25: `gh workflow run train.yml -f reason="drift PSI <value>"`.

## Performance

Alert `RecallDrop`: recall below 50 % on confirmed frauds.

Inputs can be stable (no drift alert) while the input -> fraud relationship has changed:
this is concept drift. Retrain on recent labelled data, and check the gate before promotion.

## Rollback

Target: less than 5 minutes between the decision and the return to normal.

```bash
kubectl scale deploy/fraud-api-v2 --replicas=0     # canary: cut v2
kubectl rollout undo deploy/fraud-api-v1           # in-place update: back to the previous revision
kubectl rollout status deploy/fraud-api-v1
```

Write down the decision time and the recovery time in the incident report.
