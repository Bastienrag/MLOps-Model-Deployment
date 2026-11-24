# Soutenance : 10 minutes de présentation, 10 minutes de questions (26/11/2026)

Critères du jury : qualité du code, automatisation CI/CD, suivi MLflow, utilisabilité de l'API, cohérence de
l'architecture (un modèle meilleur que le hasard suffit), collaboration sur GitHub.

## Déroulé (10 minutes)

| Minutes | Partie | À montrer |
|---|---|---|
| 0:00–1:00 | Problème et données : 5 catégories de trajets, GBG500, 500 trajets, classes déséquilibrées | README, section « Données » |
| 1:00–2:30 | Modèle : fenêtres de 256 mesures, petit CNN, évaluation par trajet, validation croisée ; **0,37 de F1 macro contre 0,18 au hasard** | tableau de la validation croisée |
| 2:30–4:00 | Reproductibilité : DVC (données), MLflow (runs, registry, alias), `imu.repro` | interface MLflow, `docs/captures/mlflow-registry.txt` |
| 4:00–5:30 | API et web app : contrat Pydantic, `/docs`, 422 détaillé, version du modèle dans chaque réponse | **démo en direct** : http://localhost:8080/app/ puis Swagger |
| 5:30–7:30 | CI/CD : CI (lint, tests, gate), image multi-architecture, déploiement par runner auto-hébergé, rollback | onglet Actions : un run vert et l'échec simulé en rouge |
| 7:30–8:30 | Kubernetes et tenue en charge : 3 replicas, probes, saturation ~30 puis ~83 req/s, P95 100 ms, coût | `docs/captures/kubernetes.txt`, tableau du test de charge |
| 8:30–9:15 | Airflow : DAG `imu_pipeline`, deux exécutions identiques | graphe du DAG ou README |
| 9:15–10:00 | Limites et suite : précision modeste (4 trajets *Reckless*), pas de Secret K8s, runner sur dépôt public ; monitoring et réentraînement non faits | — |

## Démo en direct (à préparer avant)

1. Cluster et runner en marche : `kubectl get pods` (3 pods `Running`), `gh api repos/Bastienrag/MLOps-Model-Deployment/actions/runners`
   (runner `online`). Le runner n'est pas un service : le relancer après un redémarrage du serveur
   (`cd ~/actions-runner && sg docker -c "setsid nohup ./run.sh > runner.log 2>&1 < /dev/null &"`).
2. Ouvrir http://localhost:8080/app/ (tunnel SSH si le serveur est distant : `ssh -L 8080:localhost:8080 ...`),
   choisir un trajet, prédire, montrer une entrée invalide.
3. Rollback : `gh workflow run deploy.yml -f simulate_failure=true`, puis montrer le run rouge et que
   `curl localhost:8080/health` répond encore (environ 1 min 30).
4. Plan B si le réseau ou le serveur lâche : les captures de `docs/captures/` et les runs déjà visibles sur GitHub.

## Questions probables

- **Pourquoi un si petit modèle ?** 350 trajets d'entraînement et 4 *Reckless* : un CNN plus gros était au
  hasard. La validation croisée donne la mesure fiable, val et test n'ont qu'un *Reckless* chacun.
- **Pourquoi pas l'accuracy ?** « Safe » partout donne 0,83. F1 macro et balanced accuracy comptent chaque classe pareil.
- **Pourquoi le modèle est-il dans l'image et pas chargé depuis MLflow ?** GitHub et le cluster n'atteignent
  pas le registry local ; l'image est autonome et reproductible. Le registry reste la source en dev.
- **Pourquoi un runner auto-hébergé ?** Le cluster kind est sur le serveur : un runner GitHub ne l'atteint pas.
  Risque sur dépôt public : garde-fou (CI verte, push sur `main`, même dépôt) et approbation des PR de forks.
- **Pourquoi pas de Secret Kubernetes ?** Aucun identifiant n'est nécessaire dans le pod.
- **Que se passe-t-il si le déploiement échoue ?** `kubectl rollout status` échoue, `rollout undo` rétablit
  la version précédente ; l'API reste disponible (readiness et `maxUnavailable: 0`).
- **Comment savoir quelle version tourne ?** `/health` et l'en-tête `X-Model-Version` donnent le SHA du commit.
- **Qu'est-ce qui limite le débit ?** La limite CPU de 500m par pod : ~17 ms de CPU par requête, dont
  ~8 ms de transport HTTP, moins de 1 ms d'inférence.
- **Qu'ajouter ensuite ?** Monitoring (Prometheus, Grafana), détection de dérive et réentraînement
  automatique, plus de trajets *Reckless*.
