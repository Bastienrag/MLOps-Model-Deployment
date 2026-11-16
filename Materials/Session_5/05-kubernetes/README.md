# Session 5 lab: Kubernetes and load

The fraud API from session 4 now has to run on a cluster, and you have to see how it behaves
under load. This folder gives you everything: the API code, the model (`model/`), a sample of the
golden data for the load test, and the Kubernetes manifests. Nothing to train.

Have a look at the Makefile.md to see all commands.

## Setup

Install Docker, `kind`, `kubectl` and `uv`, then:

```bash
uv sync                  # Python 3.14 and the project libraries
source .venv/bin/activate
docker info              # must succeed
```

## 1. Make sure that your API is running locally

```bash
make api
curl localhost:8000/health
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d @sample.json
```

## 2. Cluster and v1

Let's build our Kubernetes cluster. We'll be using kind (Kubernetes in Docker).

```bash
make kind-up                       # kind cluster "mlops"
make image VERSION=v1              # docker build -t fraud-api:v1 .
make kind-deploy VERSION=v1        # kind load + kubectl apply
kubectl get pods
curl http://localhost:8080/health  # kind maps host port 8080 to the Service (NodePort 30080)
```

Watch the pods go from `Pending` to `Running` with `kubectl get pods -w`.

## 3. Complete the manifests for the probes

Three TODO are left in `k8s/`:

- **TODO 1 and TODO 2** in `k8s/deployment.yaml`: a readiness probe, a liveness probe, and the resources (requests and limits). Justify your values.
- **TODO 3** in `k8s/canary.yaml`: configure the canary like v1.

Redeploy after each change with `make kind-deploy VERSION=v1`.

## 4. Load test at 1 and 3 replicas

The load test runs against the cluster on port 8080. Use the same command for both runs, so the
comparison is fair.

```bash
kubectl scale deploy/fraud-api-v1 --replicas=1
kubectl rollout status deploy/fraud-api-v1
make load-kind CSV=reports/load_k8s_1r

kubectl scale deploy/fraud-api-v1 --replicas=3
kubectl rollout status deploy/fraud-api-v1
make load-kind CSV=reports/load_k8s_3r
```

Read the `95%` column (P95 in ms, row `/predict`) and the `Requests/s` column in
`reports/load_k8s_<N>r_stats.csv`. Wait about 30 seconds after each rollout so that the pods are ready and warm.

To find the saturation point, rerun with more users, for example by changing `-u 50` to `-u 100` in
the Makefile. Watch where the P95 starts to climb.

## 5. Latency budget

The target is P95 ≤ 200 ms. Split it into four spans and fill in the table:

| Span | How to measure it |
| --- | --- |
| Total end-to-end P95 | Locust `95%` column, `/predict` row |
| Network and port mapping | Client P95 minus server P95 (below) |
| Server-side P95 | `curl -s localhost:8080/metrics \| grep api_request_duration_seconds`, or Prometheus |
| Model inference | `inference_ms` in the prediction logs (`logs/predictions.jsonl` locally) |
| Validation, serialisation, logging | Server P95 minus the inference part |

The histogram buckets are coarse around 200 ms, so the server quantile is approximate. Give it as
an order of magnitude. The P95 of each span does not add up exactly to the total.

## 6. Cost per 1,000 predictions

On kind the infrastructure is free, so the cost is an assumption. Choose an hourly price `C_h` for
one replica, and justify it from the requests and limits in `deployment.yaml`. Use the throughput at
saturation, not the offered load:

```
cost per 1,000 predictions = (C_h × replicas × 1000) / (throughput_rps × 3600)
```

Compare the 1-replica and 3-replica runs.

## 7. Canary (optional)

Build v2 from the same code, with a different threshold (set in `k8s/canary.yaml`):

```bash
docker build -t fraud-api:v2 .
kind load docker-image fraud-api:v2 --name mlops
kubectl apply -f k8s/canary.yaml
kubectl scale deploy/fraud-api-v1 --replicas=9
kubectl scale deploy/fraud-api-v2 --replicas=1      # about 10% of the traffic goes to v2
kubectl scale deploy/fraud-api-v2 --replicas=0      # rollback
```

If a pod shows `ErrImagePull`, the image is missing from the cluster: run the `docker build` and
`kind load` for that tag.

## Questions to think about

- A pod is `Running` but `0/1 Ready`. Who receives traffic, and what happens if the readiness probe is wrong?
- You added a second replica and throughput did not double. What would you look at first, and why?
- Which span dominates your budget? If you halve it, what is the new P95, and does the target become easier to meet?
- The `inference_ms` field does not cover the whole server time. What is in the gap, and does it matter?
- With 9 and 1 replicas, why is the 10% only approximate? What would make the split exact, and what would you lose?
- Suppose v2 leaks memory. Which signal shows it first, and why is the liveness probe not enough to catch it?
- Is P95 ≤ 200 ms the right target for a card payment? Who should decide, and what would you ask them?
- The cost per prediction depends on throughput. At what throughput does 1 replica become more expensive than 3?
- The load test runs through the kind port mapping and on the same laptop as the cluster. Which direction does this bias the results, and how would you check it?

## Notes

- `make load` tests the local API on port 8000. `make load-kind` tests the cluster on port 8080.
- The load test runs on the same machine as the cluster and shares its CPU. Keep that in mind when you interpret the numbers.
