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
