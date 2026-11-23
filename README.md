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
