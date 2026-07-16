# Projet 8 — Segmentation sémantique d'images pour véhicule autonome

Projet Master AI Engineer (CentraleSupelec / OpenClassrooms).  
Développement d'un pipeline de segmentation sémantique d'images urbaines pour un système embarqué de véhicule autonome, avec déploiement d'une API de démonstration sur Azure.

---

## Contexte

**Future Vision Transport** est une entreprise spécialisée dans les systèmes embarqués pour véhicules autonomes. L'objectif est de concevoir un modèle de segmentation sémantique capable d'identifier et de délimiter les différentes zones d'une scène urbaine (route, piétons, véhicules, bâtiments…) à partir d'images RGB, afin d'alimenter les systèmes de décision du véhicule.

---

## Branches Git

| Branche | Usage |
|---------|-------|
| `main` | Production — code stable |
| `develop` | Développement actif |

---

## Données

Source : [Cityscapes Dataset](https://www.cityscapes-dataset.com/) — images de conduite urbaine annotées pixel par pixel.

| Donnée | Valeur |
|--------|--------|
| Images d'entraînement | 2 975 |
| Images de validation | 500 |
| Résolution originale | 2 048 × 1 024 px |
| Résolution utilisée | 256 × 128 px |
| Classes originales | 34 |
| Classes agrégées (projet) | 8 |

### Classes retenues (agrégation Cityscapes)

| ID | Classe | Contenu | Criticité sécurité |
|----|--------|---------|-------------------|
| 0 | `void` | Non étiqueté, hors ROI | — |
| 1 | `flat` | Route, trottoir | — |
| 2 | `construction` | Bâtiments, murs, clôtures | — |
| 3 | `object` | Panneaux, poteaux, feux | Élevée |
| 4 | `nature` | Végétation, terrain | — |
| 5 | `sky` | Ciel | — |
| 6 | `human` | Piétons, cyclistes | ⚠️ Critique |
| 7 | `vehicle` | Voitures, bus, motos | ⚠️ Critique |

> Le jeu de test officiel Cityscapes n'étant pas annoté publiquement, le jeu de validation original est utilisé en tant que jeu de test. Un split `val → test` est documenté dans `data.ipynb`.

---

## Architectures testées

Quatre architectures encodeur–décodeur de type U-Net ont été comparées, selon deux stratégies d'entraînement :

- **Transfer learning (encodeur gelé)** : seul le décodeur est entraîné, les poids ImageNet sont conservés
- **Fine-tuning complet** : tout le réseau est entraîné depuis les poids ImageNet (`freeze_encoder=False`)

| Architecture | Paramètres | Encodeur | Stratégie |
|---|---|---|---|
| UNet-mini | ~0,2 M | Aucun (from scratch) | Baseline |
| MobileNetV3Small-UNet | ~1,8 M | MobileNetV3Small | Gelé + Fine-tuning |
| VGG16-UNet | ~14 M | VGG16 | Gelé |
| ResNet50-UNet | ~25 M+ | ResNet50 | Gelé + Fine-tuning |

---

## Fonction de perte

Combinaison **Dice Loss + Focal Loss** (`DiceFocalLoss`, définie dans `custom_object.py`) :

- **Dice Loss** : mesure le chevauchement entre masque prédit et masque réel (robuste au déséquilibre de classes)
- **Focal Loss** : amplifie la pénalité sur les pixels difficiles (faible confiance du modèle)
- **Pondération pixel par pixel** : chaque pixel est pondéré selon sa classe (`sample_weight`), calculé à partir de la distribution de pixels dans `data.ipynb`
- Poids appliqués **avant** la réduction (`reduce_mean`), afin que les classes rares (`human`, `object`) influencent réellement le gradient

> Le modèle ne contient **pas** d'activation softmax en sortie : la `DiceFocalLoss` applique son propre softmax en interne.

---

## Métriques d'évaluation

| Métrique | Description |
|---|---|
| **Dice** | Chevauchement moyen masque prédit / masque réel. Pas un seuil de décision — valeurs de référence Cityscapes : > 0,60 correct, > 0,75 bon |
| **mIoU** | Intersection over Union moyen sur toutes les classes |
| **Pixel Accuracy** | % de pixels correctement classés |
| **IoU par classe** | Dont `IoU_human` et `IoU_object`, indicateurs critiques pour la sécurité |

> La décision de classe par pixel repose sur un `argmax` sur les 8 probabilités softmax — pas sur un seuil Dice.

---

## Résultats

*Les résultats définitifs seront renseignés après réentraînement complet du benchmark (corrections de code appliquées).*

| Architecture | mIoU val | Dice val | IoU human val | Taille modèle |
|---|---|---|---|---|
| UNet-mini | — | — | — | — |
| MobileNetV3 (gelé) | — | — | — | — |
| MobileNetV3 (fine-tuning) | — | — | — | — |
| ResNet50 (gelé) | — | — | — | — |
| ResNet50 (fine-tuning) | — | — | — | — |

**Modèle retenu pour le déploiement :** MobileNetV3Small-UNet — meilleur compromis performance / taille pour un système embarqué.

---

## Structure du projet

```
P8_Segmentation_Images/
│
├── classe_dataset.py          # Générateur de données Keras (ImageSegmentationDataset)
├── custom_object.py           # DiceFocalLoss + DiceMetric (objets Keras custom)
├── utils_p8.py                # Fonctions utilitaires (visualisation, métriques)
│
├── models.ipynb               # Entraînements, comparaison architectures, suivi MLflow
├── data.ipynb                 # EDA, distribution des classes, calcul des poids, split val→test
│
├── data/
│   ├── train/                 # Images + masques d'entraînement (Cityscapes)
│   ├── test/                  # Images + masques de test (subset val Cityscapes)
│   └── models/                # Modèles sauvegardés (.keras)
│
├── mlruns/                    # Expériences MLflow (tracking local)
├── mlf_1/                     # Artifacts MLflow complémentaires
│
├── tests/                     # Tests unitaires (pytest)
└── requirements.in            # Dépendances du projet
```

> L'API et le frontend sont maintenus dans des dépôts indépendants (`P8_api_app` et `P8_front_app`).

---

## Déploiement Azure

L'API de démonstration et l'interface Streamlit sont déployées dans des dépôts séparés :

| Composant | Dépôt | Description |
|---|---|---|
| API Flask | `P8_api_app` | Endpoints `/health`, `/list_img`, `/select_img`, `/predict` |
| Frontend | `P8_front_app` | Interface Streamlit de démonstration |

| Ressource Azure | Détail |
|---|---|
| Groupe de ressources | `rg-projet8` |
| App Service Plan | `ASP-rgprojet8-ba7f` (B1, Sweden Central) |
| Container Registry | `acrprojet8` |
| Blob Storage | `stprojet8seg` (France Central) — modèle + images de test |

### Gestion des crédits Azure

Les apps sont **mises en pause** entre les sessions de démonstration.

```bash
# Remettre en pause
az webapp stop --name projet8-api --resource-group rg-projet8
az webapp stop --name projet8-front --resource-group rg-projet8

# Redémarrer avant démonstration
az webapp start --name projet8-api --resource-group rg-projet8
az webapp start --name projet8-front --resource-group rg-projet8
```

---

## Suivi expérimental — MLflow

Toutes les expériences sont tracées avec MLflow (local). 

- Métriques loggées par epoch :

    - `train_dice`, `val_dice` — coefficient Dice
    - `train_miou`, `val_miou` — mean IoU
    - `train_pixel_accuracy`, `val_pixel_accuracy`
    - IoU par classe : `iou_class_flat`, `iou_class_human`, `iou_class_vehicle`…

```bash
# Lancer l'interface MLflow
mlflow ui --backend-store-uri mlruns/
```

---

## Reproductibilité

Seeds fixées en début de notebook :

```python
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
```

---

## Environnement Python

Python 3.10+. Dépendances principales :

```
tensorflow==2.16.2
keras==3.10.0
numpy==1.26.4
pandas
scikit-learn
Pillow==11.1.0
albumentations==2.0.8
opencv-python-headless
matplotlib==3.10.0
seaborn
mlflow
pytest==8.4.1
```

> L'API (`P8_api_app`) et le frontend (`P8_front_app`) ont leurs propres `requirements.txt`.

```bash
pip install -r requirements.txt
```
