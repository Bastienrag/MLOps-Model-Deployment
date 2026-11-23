# MLOps-Model-Deployment

## Données du projet

Les données ne sont pas versionnées dans ce dépôt : elles sont trop lourdes pour GitHub
(`SRV3490.parquet` dépasse 250 Mo). Chaque membre du groupe les garde en local.

Pour les récupérer :

1. Télécharger `project.zip` sur Inside :
   **MLOps & Model Deployment > Materials > Project > project.zip**.
2. Le décompresser dans un dossier en dehors du dépôt, par exemple
   `~/Desktop/Data for project MLOps/`.

| Fichier | Contenu |
|---|---|
| `GBG500.parquet` | Données IMU de 500 trajets à Göteborg, répartis en 5 catégories (Safe, Safe Overall, Bad Weather, Tandem, Reckless) |
| `GBG500_labels.csv` | Labels de GBG500, réservés à l'évaluation |
| `SRV3490.parquet` | Données IMU de 3490 trajets dans 8 villes européennes, dont 698 avec une chute confirmée |

Le détail des jeux de données est dans [`Project/data.txt`](Project/data.txt).

## Environnement de développement local

Une seule commande lance MLflow (suivi des expériences et registre de modèles) et un
stockage S3 où sont rangés les artefacts des modèles. Prérequis : Docker et Docker Compose.

```bash
cp .env.example .env     # puis choisir les mots de passe dans .env (jamais commité)
docker compose up -d
```

| Service | URL |
|---|---|
| MLflow | http://localhost:5000 |
| API S3 (DVC, MLflow, boto3) | http://localhost:8333 |
| Navigateur de fichiers S3 | http://localhost:8888/buckets/ |

Deux buckets sont créés au premier démarrage : `mlflow` (artefacts des modèles) et `dvc`
(remote des données). Les clés d'accès S3 et le mot de passe Postgres viennent du fichier
`.env` ; ils ne sont pas dans les images. Si le port 5000 est déjà pris sur la machine,
décommenter `MLFLOW_PORT` dans `.env`.

Le stockage S3 est SeaweedFS, comme dans les labs : l'image Docker officielle de MinIO
n'est plus publiée.

Arrêt : `docker compose down` (les données restent), `docker compose down -v` pour tout effacer.

## Données versionnées et prétraitement (DVC)

Les données brutes GBG500 et le résultat du prétraitement sont versionnés avec DVC : Git
garde les fichiers `.dvc` et `dvc.lock` (des empreintes), les données sont sur le remote S3
(bucket `dvc` de l'environnement de dev). Un commit désigne donc le code et les données exactes.

```bash
uv sync                                    # installe les dépendances (uv.lock)
docker compose up -d                       # le remote DVC est le S3 local
uv run dvc remote modify --local storage access_key_id <S3_ACCESS_KEY>
uv run dvc remote modify --local storage secret_access_key <S3_SECRET_KEY>
uv run dvc pull                            # récupère données brutes et jeux prétraités
```

Les clés sont celles de `.env`. Pour reconstruire les jeux depuis les données brutes :
`uv run dvc repro`, puis `uv run dvc push` après un changement.

Le prétraitement (`src/imu/data.py`, paramètres dans `params.yaml`) :

- **Valeurs manquantes** : les lignes dont les 6 capteurs sont NaN (0,7 %, séquences de 19
  points au plus) sont interpolées linéairement.
- **`time_index` en double** : un trajet a deux mesures au même instant ; on garde la première.
- **Découpage** : 70 / 15 / 15 par trajet, stratifié par classe, graine fixée. Chaque jeu
  contient au moins un trajet Reckless (4 / 1 / 1), donc les métriques de validation et de test
  sont bruitées sur cette classe.
- **Fenêtrage** : fenêtres de 256 points, pas de 128, un label par fenêtre (celui du trajet).
  Les trajets font de 1 807 à 4 767 points : le fenêtrage multiplie les exemples
  (7 791 / 1 715 / 1 693 fenêtres) sans padding. Le découpage se fait avant le fenêtrage, donc
  aucun trajet n'a de fenêtres dans deux jeux. La prédiction d'un trajet agrège celles de ses fenêtres.
- **Normalisation** : moyenne et écart-type par capteur calculés sur le train seul
  (`data/processed/meta.json`).

## Modèle et évaluation

```bash
uv run dvc repro        # prétraitement, baselines, CNN, validation croisée
uv run python -m imu.train   # le CNN seul : models/cnn.pt et reports/cnn.json
```

- **Baselines** (`src/imu/baseline.py`) : classe majoritaire, et régression logistique pondérée
  sur 30 statistiques par fenêtre (moyenne, écart-type, min, max, variation de chaque capteur).
- **Modèle** (`src/imu/model.py`) : CNN 1D de 3 couches (8, 16, 16 canaux), pooling moyenne + max
  dans le temps, 5 sorties. Le réseau est petit à dessein : 350 trajets d'entraînement seulement.
  Un trajet est classé en moyennant les probabilités de ses fenêtres.
- **Déséquilibre** : perte pondérée par l'inverse de l'effectif de chaque classe, et
  augmentation par bruit et variation d'échelle. Métriques : F1 macro, balanced accuracy, matrice
  de confusion et rappel par classe (jamais l'accuracy seule : prédire « Safe » partout donne 0,83).
- **Reproductibilité** : graines et nombre de threads fixés, deux entraînements donnent le même
  `models/cnn.pt` à l'octet près.

**Validation croisée** (`src/imu/cv.py`) : 5 plis stratifiés par trajet sur train + val (424 trajets,
dont 5 Reckless). Les jeux de validation et de test ne contiennent qu'un trajet Reckless chacun :
leurs scores varient beaucoup d'un tirage à l'autre, la validation croisée est la mesure à retenir.

| Modèle | F1 macro | Balanced accuracy |
|---|---|---|
| Hasard / classe majoritaire | 0,18 | 0,20 |
| Régression logistique | 0,32 | 0,50 |
| CNN | 0,37 | 0,43 |

Le CNN est nettement au-dessus du hasard. Il ne dépasse que de peu la régression logistique
(meilleur en F1 macro, moins bon en balanced accuracy) : les classes Tandem et Bad weather sont mal
reconnues. Les scores sur le jeu de test (`reports/cnn.json`) sont donnés à titre indicatif.

## Suivi MLflow et registry

Avec l'environnement de dev lancé (`docker compose up -d`) :

```bash
export MLFLOW_PORT=5000                        # ou 5001 si MLFLOW_PORT est réglé dans .env
uv run python -m imu.track                     # entraîne et trace un run, affiche son RUN_ID
uv run python -m imu.register RUN_ID staging   # met le modèle dans le registry avec l'alias staging
uv run python -m imu.score staging             # charge models:/imu-classifier@staging et le score
uv run python -m imu.register RUN_ID production   # promotion
```

Chaque run journalise :

- les **paramètres** (`data.*`, `train.*`) et les **métriques** (F1 macro, balanced accuracy, rappel par
  classe, par trajet et par fenêtre, sur val et test), plus la perte et le F1 de validation par époque ;
- les **tags** `git_commit`, `git_dirty` (vrai si le dépôt n'est pas propre : le run n'est alors pas
  reproductible à coup sûr) et les empreintes des données `data_raw_md5`, `data_labels_md5`,
  `data_processed_md5` ;
- le **modèle** (signature : fenêtres `(n, 256, 6)` en entrée, probabilités `(n, 5)` en sortie, softmax
  inclus), et les artefacts `meta.json` (classes, moyenne et écart-type de normalisation), `params.yaml`
  et `baseline.json`. Les fichiers sont dans le bucket S3 `mlflow`.

Le modèle se charge par nom et alias (`models:/imu-classifier@staging`), jamais par chemin de fichier.

**Reproduire un run depuis un clone** : relever le `git_commit` du run dans l'interface MLflow, puis

```bash
git clone <dépôt> && cd <dépôt> && git checkout <git_commit>
uv sync --locked
uv run dvc pull                                # les données de ce commit
uv run python -m imu.repro RUN_ID              # réentraîne et compare à 1e-6
```

`imu.repro` vérifie que le code et les données sont ceux du run, réentraîne avec les paramètres du run
et compare chaque métrique ; il affiche `REPRODUIT` ou `NON REPRODUCTIBLE`.

Les artefacts passent par le serveur MLflow (`--serve-artifacts`) : le client n'a besoin d'aucune clé S3.
Il désactive le téléchargement par URL présignée, qui pointerait vers `s3:8333`, une adresse interne à Docker.

## API de prédiction

L'API charge une fois, au démarrage, le modèle `models:/imu-classifier@production` du registry MLflow
(`MODEL_URI` pour un autre alias) et les statistiques de normalisation enregistrées avec lui.

```bash
export MLFLOW_PORT=5001                        # le serveur MLflow doit tourner (docker compose up -d)
uv run uvicorn imu.api.main:app --port 8000
```

- Documentation Swagger : http://localhost:8000/docs (l'exemple de requête est un trajet valide :
  « Try it out » fonctionne directement).
- `POST /v1/predict` : un trajet (`samples`, de 256 à 10 000 mesures `ax ay az rx ry rz`, au pas de
  100 ms, et un `ride_id` facultatif) -> `predicted_class`, `probability`, `probabilities` des 5 classes,
  `n_windows` et `model_version`. Une ligne manquante s'envoie avec ses 6 valeurs à `null` : elle est
  interpolée comme à l'entraînement.
- `GET /health` : `200` si le modèle est chargé, `503` sinon. La prédiction répond aussi `503` sans
  modèle, et `422` avec le détail des champs en cause pour une entrée invalide (hors bornes, champ
  inconnu, trajet trop court, ligne à moitié vide).
- Chaque réponse porte la version du modèle (champ `model_version` et en-tête `X-Model-Version`).

Le prétraitement de l'API est celui de l'entraînement (mêmes fonctions de `imu.data`) : pour un même
trajet, l'API renvoie exactement les probabilités de l'évaluation hors ligne.

Les tests (`uv run pytest`) injectent un faux modèle : ils tournent sans serveur MLflow.

## Web app de test

Une page web, servie par l'API elle-même, permet d'essayer le modèle sans écrire de requête.
Lancer l'API (voir ci-dessus), puis ouvrir **http://localhost:8000/app/**.

1. **Choisir un trajet** : un des 5 trajets d'exemple (un par catégorie, extraits du jeu de test
   GBG500, 600 mesures chacun), ou charger un fichier. Le fichier est soit un CSV avec les colonnes
   `ax ay az rx ry rz` (une cellule vide est une mesure manquante, les autres colonnes sont ignorées),
   soit un JSON : une liste de mesures, ou un objet avec une clé `samples`. Les signaux du trajet
   s'affichent (accélération en haut, gyroscope en bas).
2. **Prédire** : la page appelle `POST /v1/predict` et affiche la catégorie, sa probabilité et les
   probabilités des 5 classes. Pour un trajet d'exemple, la page rappelle la vraie catégorie.

Les erreurs de l'API sont affichées en clair : une entrée invalide (422) liste les champs en cause
(`samples[5].ry : …`), un modèle non chargé (503) indique que le service est indisponible, et une API
injoignable est signalée. Le bandeau en haut montre l'état de `/health` et la version du modèle.

Sur ces extraits d'une minute, le modèle ne classe pas tous les exemples correctement (c'est un
modèle modeste, voir l'évaluation) : la page sert à montrer le fonctionnement de la chaîne, pas à
promettre une précision. Les exemples se régénèrent avec `uv run python -m imu.web.make_examples`
(nécessite les données, `dvc pull`).

## Intégration continue (GitHub Actions)

`.github/workflows/ci.yml` s'exécute à chaque push et pull request :

1. **`lint-test`** : `uv sync --locked`, `ruff check`, `pytest`.
2. **`model-gate`** (après `lint-test`) : évalue le modèle sur un jeu figé et échoue sous un seuil.

Le gate (`uv run python -m imu.gate`) évalue `models/cnn.pt` sur `ci/golden.npz`, les 76 trajets du jeu
de test (une fenêtre sur deux, 2,5 Mo). Il échoue (code 1, job rouge) si la balanced accuracy par trajet
est sous 0,30 ou le F1 macro sous 0,25 (`gate` dans `params.yaml` ; le hasard donne 0,20). Le modèle
actuel obtient 0,55 et 0,42. Le rapport est téléchargeable dans les artefacts du run
(`gate-report`) et résumé dans la page du run.

Les runners GitHub n'atteignent ni le remote DVC ni le registry (ils sont locaux). Le modèle (15 Ko) et le
jeu figé sont donc versionnés dans Git, produits par les étapes `train` et `golden` du pipeline DVC
(`cache: false`). Après un réentraînement : `dvc repro`, puis commiter `models/cnn.pt`, `ci/golden.npz` et
`dvc.lock`.

Bonnes pratiques appliquées : actions épinglées par empreinte de commit, `permissions: contents: read`,
`timeout-minutes` sur chaque job, annulation des exécutions obsolètes (`concurrency`), version d'uv fixée.
Aucun job ne publie quoi que ce soit : l'image Docker n'est construite et poussée qu'après la réussite des
deux jobs.

**Vérifier qu'un échec met la pipeline au rouge** (échec volontaire, sans toucher au code) :

```bash
gh workflow run ci.yml -f gate_min_balanced_accuracy=0.99
gh run watch
```

En local : `uv run python -m imu.gate`, avec `GATE_MIN_BALANCED_ACCURACY=0.99` pour provoquer l'échec.

## Image Docker de l'API

```bash
docker build --build-arg GIT_SHA=$(git rev-parse HEAD) -t imu-api:dev .
docker run --rm -p 8000:8000 imu-api:dev          # http://localhost:8000/app/
```

L'image embarque le modèle (`models/cnn.pt` et `models/meta.json`) : elle n'a besoin ni de MLflow, ni du
S3, ni d'aucun identifiant. Elle charge le modèle depuis ces fichiers (`MODEL_URI=/app/models`) au lieu du
registry. La version annoncée par l'API dans chaque réponse est le commit qui a produit l'image
(`GIT_SHA`).

- **Dépendances** : `uv sync --locked --no-default-groups`, donc les versions exactes de `uv.lock`, sans
  MLflow, DVC, scikit-learn ni outils de dev (ils sont dans les groupes `mlops` et `dev` de
  `pyproject.toml`). L'image fait 1,5 Go, dont 690 Mo pour PyTorch (version CPU).
- **Cache de build** : les dépendances sont installées avant la copie du code, puis le code, puis le
  modèle. Un changement de code ne réinstalle pas PyTorch.
- **Sécurité** : le conteneur tourne avec un utilisateur sans privilèges (uid 10001). `.dockerignore` est
  une liste blanche (`pyproject.toml`, `uv.lock`, `src`, `models/cnn.pt`, `models/meta.json`) : ni `.env`,
  ni données, ni `.git` n'entrent dans le contexte de build.
- **Smoke test** (`ci/smoke_test.py`) : démarre l'image, vérifie `/health`, la version du modèle, une
  prédiction, les probabilités et le refus d'une entrée invalide en 422. Il tourne en local et en CI.

**Publication.** Le job `docker` de la CI construit l'image après le lint, les tests et le gate. Sur un
push vers `main`, une fois le smoke test réussi, il la pousse sur Docker Hub avec deux tags : le commit
(`<compte>/imu-api:<sha complet>`) et `latest`. Réglages du dépôt, à faire une fois :

```bash
gh variable set DOCKERHUB_USERNAME --body basutsu
gh secret set DOCKERHUB_TOKEN        # token d'accès Docker Hub (Account settings > Personal access tokens)
```

Sans ces réglages, le job construit et teste l'image sans la publier.

**Image publiée** : https://hub.docker.com/r/basutsu/imu-api

```bash
docker run --rm -p 8000:8000 basutsu/imu-api:latest      # ou un tag précis : basutsu/imu-api:<sha du commit>
```

Le tag est le commit complet qui a produit l'image ; c'est aussi la version du modèle renvoyée par l'API.
L'image est construite par la CI sur une architecture **amd64** : sur un Mac Apple Silicon ou une machine
arm64, Docker l'exécute par émulation, ou on la reconstruit localement avec `docker build` (voir plus haut).
