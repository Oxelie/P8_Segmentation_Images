# Contenu de la présentation — Projet 8
## "Traiter les images pour le système embarqué de voitures autonomes"

**Master AI Engineer — OpenClassrooms**  
Stéphanie Duhem — Juillet 2026

---

## Slide 1 — Titre

**SOUTENANCE PROJET 8**  
"Traiter les images pour le système embarqué de voitures autonomes"

Parcours AI Engineer — OpenClassrooms  
Stéphanie Duhem · Juillet 2026

---

## Slide 2 — 00. Contexte & Objectifs

### Le contexte

**"Future Vision Transport"** : Entreprise spécialisée en systèmes embarqués de vision pour véhicules autonomes.  
Mon rôle : bloc segmentation d'images dans la pipeline R&D.

### Contraintes techniques

| Contrainte | Détail |
|---|---|
| ⚡ Rapidité | Prédiction en temps réel pour le système embarqué |
| 🎯 Qualité | 8 classes dont des objets critiques (piétons) |
| 🔧 Framework | Keras imposé pour la cohérence de l'équipe |
| ☁ Déploiement | API & App web sur le Cloud |
| 🚸 Data | Classes minoritaires critiques (humains, objets) |

### Objectifs

1. **1 modèle** : Entraîner pour identifier 8 catégories Cityscapes
2. **1 API** : Déployer sur le Cloud pour les prédictions
3. **1 Application Web** : Pour visualiser les résultats

---

## Slide 3 — 01-A. Dataset Cityscapes — Vue globale

### Contexte scénographique

- Allemagne, circulation citadine, prises depuis l'avant d'une Mercedes
- Images diurnes, météos variant de beau temps à couvert

**Axe d'enrichissement des données (perspectives) :**  
Images nocturnes, autres villes, autre caméra, autres paysages (campagne, autoroute, pays), météo plus variée (brouillard, pluie…)

| Donnée | Valeur |
|---|---|
| Images d'entraînement | 2 975 |
| Images de validation | 500 |
| Résolution originale | 2 048 × 1 024 px |
| Résolution utilisée | 256 × 512 px |
| Classes originales | 34 |
| Classes agrégées | 8 |

| ID | Classe | Criticité |
|---|---|---|
| 0 | `void` | — |
| 1 | `flat` (route, trottoir) | — |
| 2 | `construction` (bâtiments) | — |
| 3 | `object` (panneaux, feux) | ⚠ Élevée |
| 4 | `nature` (végétation) | — |
| 5 | `sky` | — |
| 6 | `human` (piétons, cyclistes) | ⚠ Critique |
| 7 | `vehicle` (voitures, bus) | ⚠ Critique |

---

## Slide 4 — 01-B. L'enjeu central des classes déséquilibrées

| Étape | Détail |
|---|---|
| **Problème** | `human` + `object` = 3 % des pixels → modèle naïf les ignore |
| **Solution 1** | Pondération des 8 classes (Inverse Frequency), intégrée dès le batch |
| **Solution 2** | Loss combinée Dice + Focal, pour focus sur les classes difficiles |
| **Impact** | Meilleur rappel sur piétons et signalétique → sécurité du modèle |

---

## Slide 5 — 02. Panorama des approches de segmentation

> Segmentation sémantique, d'instance et panoptique

**Avant 2015**  
Seuillage (Otsu 1979), clustering (K-means — Lloyd 1982), détection de contours (Canny 1986)

**Deep Learning : CNN encodeur-décodeur & Backbones pré-entraînés (2015–2020)**  
- U-Net (skip connections), FPN, DeepLabV3+
- Backbones pré-entraînés dès 2015 : SegNet avec VGG16, FPN avec ResNet
- DeepLabV3 (2017) sans décodeur → DeepLabV3+ (2018) avec décodeur

**Optimisation pour déploiement embarqué (2018–2021)**  
MobileNetV2 (2018), MobileNetV3 (2019), EfficientNet (2019)

**Transformer & Foundation Models (2021+)**  
ViT (2020) → SegFormer (2021), CLIP (2021), SAM (2023)  
Zero-shot learning (généralisation multi-tâches) avec CLIP ou SAM

**Notre choix :** Architectures U-Net avec backbones pré-entraînés — bon compromis précision / coût / contraintes embarquées

---

## Slide 6 — 03. Pipeline de données industrialisable

### La classe `ImageSegmentationDataset`

| Apport | Détail |
|---|---|
| **Réutilisable** | Héritage de `keras.utils.PyDataset` → compatible avec `model.fit()` et MLflow |
| **Augmentations** | Albumentations intégré : flip horizontal, variations luminosité/contraste sur la paire image-masque |
| **Poids** | Pondération par pixel dès le batch (Inverse Frequency) — triplet (image, masque, poids) intégré dans la loss |
| **Multi-modèles** | Prétraitement spécifique à chaque backbone |

**Pipeline :**  
`Chargement (image + masque)` → `Mapping 34 → 8 classes` → `Normalisation & prétraitement` → `Augmentation (opt.)` → `Pondération pixels` → `Batch TensorFlow`

---

## Slide 7 — 04-A. Benchmark — Architectures U-Net

| Modèle | Paramètres | Taille | Entraînement | Usage dans le projet |
|---|---|---|---|---|
| UNet-mini | ~0,19 M | ~2 Mo | From scratch | BASELINE |
| MobileNetV3Small-UNet | ~1,5–2 M | ~17 Mo | Transfer Learning → Fine-tuning | Candidat naturel |
| VGG16-UNet | ~15–20 M | ~119 Mo | ImageNet | Challenger |
| ResNet50-UNet | ~25 M | ~494 Mo | Transfer Learning → Fine-tuning | Upper bound |

---

## Slide 8 — 04-B. Deux stratégies d'entraînement comparées

### Transfer learning — encodeur gelé

- Encodeur (backbone) : poids ImageNet **gelés**
- Décodeur : seul le décodeur U-Net est entraîné
- ✓ Convergence rapide
- ✓ Stable (BatchNorm stable — stats ImageNet valides)
- − Représentations encodeur non adaptées à Cityscapes

### Fine-tuning complet + Data Augmentation

- Encodeur + Décodeur dégelés : tout le réseau entraîné dès l'epoch 1
- Enrichissement du dataset par la data augmentation
- ✓ Adaptation complète à Cityscapes
- ✓ Potentiel de performance supérieur
- − Plus long à converger
- − Risque d'instabilité BatchNorm (ReduceLROnPlateau + early stopping aident à traverser cette phase)

---

## Slide 9 — 04-C. Contrôle de l'apprentissage & suivi expérimental

### Keras callbacks

| Callback | Rôle |
|---|---|
| **EarlyStopping** | Arrêt si `val_loss` stagne (patience configurable) → évite le sur-apprentissage |
| **ModelCheckpoint** | Sauvegarde du meilleur modèle (val_dice) → reproductibilité garantie |
| **ReduceLROnPlateau** | LR divisé si plateau de `val_loss` détecté → convergence plus fine |

### Suivi expérimental MLflow

- **Métriques loggées** : Dice, mIoU, Pixel Accuracy par epoch (train + val)
- **Paramètres** : Architecture, LR, batch size, `freeze_encoder`, augmentations
- **Artefacts** : Meilleur modèle `.keras` + courbes d'apprentissage
- **Reproductibilité** : Seeds fixées (`PYTHONHASHSEED`, numpy, TensorFlow) + assertion

> Chaque run est traçable et comparable. La reproductibilité des résultats est garantie.

---

## Slide 10 — 05-A. Comparaison quantitative

| Architecture | Params | mIoU val | Dice val | IoU human | IoU vehicle | Taille | Inférence/img |
|---|---|---|---|---|---|---|---|
| U-Net mini | ~0,2 M | 0.349 | 0.430 | 0.011 | 0.202 | 2 Mo | 0,18 sec |
| MobileNetV3 (gelé) | ~1,8 M | 0.601 | 0.639 | 0.328 | 0.690 | 9,7 Mo | 0,29 sec |
| VGG16 (gelé) | ~15 M | 0.562 | 0.604 | 0.385 | 0.000 ⚠️ | 119 Mo | 0,21 sec |
| ResNet50 (gelé) | ~25 M+ | 0.742 | 0.768 | 0.610 | 0.841 | 314 Mo | 0,39 sec |
| **MobileNetV3 (fine-t.) ✅** | ~1,8 M | **0.661** | **0.696** | 0.418 | 0.749 | 17 Mo | 0,32 sec |
| **ResNet50 (fine-t.) 🌟** | ~25 M+ | **0.762** | **0.789** | **0.656** | **0.855** | 494 Mo | 0,38 sec |
| MobileNetV3 (fine-t. + augm.) | ~1,8 M | 0.652 | 0.691 | 0.385 | 0.747 | 17 Mo | 0,30 sec |
| ResNet50 (fine-t. + augm.) | ~25 M+ | 0.760 | 0.787 | 0.639 | 0.854 | 494 Mo | 0,38 sec |

> ⚠️ VGG16 écarté : IoU vehicle = 0.000 sur tous les splits (effondrement total sur la classe vehicle, anomalie non résolue).

📌 **Repère Dice** — Métrique de qualité globale. Cityscapes : > 0,60 correct · > 0,75 bon

**Modèle déployé :** MobileNetV3Small-UNet fine-tuning (~17 Mo, Dice 0.696) — compromis performance/taille/latence pour CPU Azure B1.

---

## Slide 11 — 05-B. Analyse qualitative

Comparaison visuelle des prédictions (masque réel vs masque prédit) :

- **MobileNetV3 fine-tuned** — 50 epochs — 0,32 sec/image
- **ResNet50 fine-tuned** — 50 epochs — 0,38 sec/image

> *(Voir PDF de présentation pour les exemples visuels côte à côte)*

---

## Slide 12 — 05-C. Modèle retenu — MobileNetV3 fine-tuned

> *(Voir PDF de présentation pour les courbes d'apprentissage)*

Modèle sélectionné pour le déploiement : **MobileNetV3Small-UNet fine-tuning**  
- val Dice = 0.696 · val mIoU = 0.661
- Taille : ~17 Mo · Inférence : ~0,32 sec/image sur CPU B1
- Candidat naturel pour un déploiement embarqué contraint

---

## Slide 13 — 06-A. Architecture de déploiement — Azure Cloud

| Composant | Détail |
|---|---|
| **Conteneurisation** | Docker (`python:3.10-slim`) — build via Azure Container Registry (`acrprojet8`) |
| **Plan Azure** | App Service B1 — 1 vCPU · 1,75 GB RAM · ~13 $/mois |
| **Lazy loading** | Modèle chargé à la première requête et conservé en mémoire pour toutes les suivantes |
| **/health** | Endpoint de santé → démarrage instantané, pas de timeout Azure |
| **API_URL** | Variable d'environnement → frontend découplé du backend |

**Ressources Azure :**
- Groupe de ressources : `rg-projet8`
- App Service Plan : `ASP-rgprojet8-ba7f` (B1, Sweden Central)
- Container Registry : `acrprojet8`
- Blob Storage : `stprojet8seg` (France Central) — modèle `.keras` + 11 images de test

**URLs de démonstration :**
- API : `https://projet8-api-e8a2apgjhhehekc4.swedencentral-01.azurewebsites.net`
- App : `https://projet8-front.azurewebsites.net`

> ⚠️ Les apps sont arrêtées entre les démonstrations (coût Azure). Redémarrage en < 1 min avant démo.

---

## Slide 14 — 06-B. Application de présentation des résultats — Fonctionnalités

### Interface Streamlit déployée

| Étape | Détail |
|---|---|
| **Liste des images** | Affichage du nom des 11 images de test disponibles — chargées depuis Azure Blob Storage via `/preload`, mises en cache (`st.cache_data`) |
| **Sélection** | Choix d'une image dans le menu → appel à l'API `/select_img` → affichage côte à côte de l'image originale et du masque réel |
| **Prédiction** | Bouton "PREDICT" → appel à l'API `/predict` → inférence du modèle MobileNetV3Small-UNet sur l'image sélectionnée (+ Spinner d'attente) |
| **Visualisation** | Retour du masque prédit en base64, accompagné du nom du modèle, de son score Dice et mIoU |

---

## Slide 15 — 06-B. Application — Suite

> *(Voir PDF de présentation pour les captures d'écran de l'interface)*

---

## Slide 16 — 07-A. Bilan du projet

### Réalisations ✓

- Pipeline ML complet et reproductible (seeds fixées)
- Classe `ImageSegmentationDataset` : générateur de données Keras industrialisable
- 4 architectures comparées avec suivi MLflow
- API Flask déployée sur Azure (endpoints `/health`, `/preload`, `/select_img`, `/predict`)
- Application Streamlit déployée et fonctionnelle
- Code versionné sur GitHub (branches `dev` / `main`) — 3 dépôts séparés : `P8_Segmentation_Images`, `P8_api_app`, `P8_front_app`
- Note technique de 10 pages

### Limites identifiées ⚠

| Limite | Détail |
|---|---|
| **Embarqué** | Le meilleur modèle, ResNet50 (~494 Mo), est inadapté à un déploiement matériel strict |
| **Démo** | MobileNetV3Small (~17 Mo) déployé — compromis vitesse/performance sur CPU B1 |
| **Données** | Cityscapes offre peu de variété (pas de pluie, nuit, brouillard) |
| **Runs** | Comparaisons limitées par les ressources de calcul |

---

## Slide 17 — 07-B. Perspectives d'amélioration

### A — Données & robustesse

- Dataset Foggy / Rain Cityscapes pour conditions dégradées
- Stratégie de pondération : test Median Frequency Balancing vs Inverse Frequency
- Augmentations géométriques plus agressives

### B — Modèle & performance

- Dégel progressif de l'encodeur (layer-wise unfreezing)
- Conversion en TFLite avec quantification INT8 pour réduire taille et latence
- Automatisation du déploiement via pipeline CI/CD (GitHub Actions)

### C — Applicatif & système

- Segmentation d'instance ou panoptique (compter les piétons)
- Connexion à un module de décision prototype
- Diminution du temps d'inférence avec GPU dédié ou ONNX Runtime

---

## Slide 18 — Conclusion

> **Merci de votre attention**

Stéphanie Duhem · Master AI Engineer — OpenClassrooms

- **API** : `projet8-api-e8a2apgjhhehekc4.swedencentral-01.azurewebsites.net`
- **App** : `projet8-front.azurewebsites.net`

---

*Document généré à partir de la présentation PPTX — Juillet 2026*
