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
