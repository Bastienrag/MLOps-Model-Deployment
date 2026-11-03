# MLOps & Model Deployment

> From notebook prototype to deployed AI system

| Semester | Track | Assessment |
|---|---|---|
| S9 | Production-readiness floor | Project + exam (50 / 50) |

## Course description

MLOps & Model Deployment converts a model-building data scientist into a colleague who deploys models as a production service. Students package a trained artefact, version data and code for reproducibility, build CI/CD pipelines with automated evaluation gates, serve behind FastAPI or BentoML, instrument the four-layer monitoring stack, detect drift, and run the canonical deployment-and-rollback patterns. It is the operational layer that distinguishes a notebook prototype from a deployed AI system, and is load-bearing for MSA-DAT09-05, MSA-DAT09-07, and the MSA-DAT09-08 capstone.

## Key topics

1. **Reproducibility stack**: versioning code (Git), data (DVC, immutable snapshots), and models (MLflow Model Registry, Hugging Face Hub); Docker packaging and the reproducibility checklist.
2. **CI/CD for ML**: automated training trigger, model-evaluation gate against a golden set, staged deployment, manual production approval, and CI cost-control discipline.
3. **Model serving**: synchronous REST with FastAPI and Pydantic validation, high-throughput batch with BentoML micro-batching, streaming inference, and the end-to-end latency budget.
4. **Four-layer monitoring**: infrastructure (p95 latency, error rate), input distribution, prediction distribution, and outcome metrics; Prometheus + Grafana instrumentation, structured logging, alerts, and runbooks.
5. **Drift detection**: data, concept, and label drift; Kolmogorov-Smirnov, chi-squared, and Population Stability Index tests wired to alerting and retraining.
6. **Retraining policy**: scheduled, drift-triggered, and performance-triggered cadences and the hybrid default.
7. **Deployment patterns**: canary, A/B, shadow mode, multi-armed bandits, and the rollback discipline with a time-to-recovery target.

## Learning outcomes

By the end of this course, students will be able to:

1. Package a trained model and its preprocessing into a reproducible artefact with MLflow, BentoML, or Docker, and identify the canonical packaging anti-patterns.
2. Version data, code, and models with DVC, MLflow Model Registry, or Hugging Face Hub, and reproduce a specific run from its commit hash and dataset version.
3. Build a CI/CD pipeline for ML with GitHub Actions: automated training trigger, model-evaluation gate, staged deployment, and manual production approval.
4. Serve a model as a REST endpoint with FastAPI or a high-throughput batch service with BentoML, implementing Pydantic input validation, request logging, and health checks.
5. Decompose an end-to-end latency budget and choose the serving pattern (synchronous, batch, streaming) and cost profile that fits a stated requirement.
6. Instrument a deployed model across the four monitoring layers (infrastructure, input distribution, prediction distribution, and business outcome) with Prometheus, Grafana, and structured logging.
7. Detect data and concept drift with the right statistical test (Kolmogorov-Smirnov, chi-squared, Population Stability Index) and tie detection to an alerting and retraining workflow.
8. Design a retraining cadence (scheduled, drift-triggered, performance-triggered) and justify the hybrid default for a stated use case.
9. Implement the canonical deployment patterns (canary, A/B, shadow) and execute a rollback within the time-to-recovery target.

## Track placement

MLOps & Model Deployment is the S9 production-readiness floor every MSA graduate must clear regardless of tech track: the ability to deploy and operate an ML model in production. It is the load-bearing upstream for **MSA-DAT09-05** Advanced Cloud AI Infrastructure, **MSA-DAT09-07** Operating AI Systems at Scale, and the **MSA-DAT09-08** Albertnative Project, all of which assume students can already version, package, monitor, and roll back an ML system end-to-end.

## Prerequisites

Required Albert courses:

| Code | Content |
|---|---|
| MSA-DATI07-01 | Git, GitHub Actions CI, virtual environments, and the Python quality toolchain (Intermediate and Advanced levels) |
| MSA-DATB07-03 | Git, GitHub Actions CI, virtual environments, and the Python quality toolchain (Entry level) |
| MSA-DATI07-05 | Leakage-safe Pipelines, paired CV, calibration, and the metric decision tree (Intermediate and Advanced levels) |
| MSA-DATB07-08 | A model trained in a leakage-safe Pipeline (Entry level) |

## Grading breakdown

| Component | Weight | Kind | Duration |
|---|---|---|---|
| Mutualised project + oral defence | 50% | CC2 | 15 min |
| Written exam | 50% | FINAL | 60 min |
| **Total** | **100%** | | |

**Written exam**: No AI. Critiquing an AI-generated plan with deliberate flaws, an applied concept check, and a case study calling for trade-off reasoning.

### Mutualised project: "Production ML System"

Shared with MSA-DAT09-02 (Data Engineering & Cloud Infrastructure).

- **Context**: data/ML engineering team at a European e-commerce company building demand forecasting.
- **MLOps deliverable**: MLflow experiment tracking, model and dataset versioning, FastAPI REST API with integration tests, Evidently-based drift monitoring with automated alerts, retraining trigger definition, rollback procedure, and simulated incident report.
- **Format**: 15 min oral (10 min presentation + 5 min Q&A). Git commit history required.

| Component | Weight |
|---|---|
| Technical deliverable through Git | 40% |
| Group oral defence | 40% |
| Individual deliberation | 10% |
| Peer review | 10% |
