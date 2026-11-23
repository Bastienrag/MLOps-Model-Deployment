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
