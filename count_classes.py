import numpy as np
import pathlib
import pickle
from collections import Counter
import os
from datetime import datetime
import argparse
from classe_dataset import ImageSegmentationDataset

#compatge des classes au niveau des pixels (= coûteux)
def count_classes_for_index(i, full_train_datagen):
    _, mask, _ = full_train_datagen.get_image_and_mask(i)
    unique, counts = np.unique(mask, return_counts=True)
    return dict(zip(unique, counts))

# compatge des classes dès la première occurrence (plus rapide, mais moins précis si une classe est très peu représentée)
# On compte chaque classe présente dans l'image comme 1, sans tenir compte de la quantité de pixels présents dans chaque classe.
# Cela permet de savoir quelles classes sont présentes dans l'image, mais pas leur fréquence.
# pas précis pour établir les poids en cas de classes déséquilibrées
# def count_classes_for_index(i, full_train_datagen):
#     _, mask, _ = full_train_datagen.get_image_and_mask(i)
#     unique = np.unique(mask)
#     # Chaque classe présente dans l'image compte pour 1
#     return dict((int(cls), 1) for cls in unique)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compte les pixels par classes d'un dataset de segmentation.")
    parser.add_argument(
        "--dataset_pickle",
        type=str,
        required=True,
        help="Chemin du fichier pickle contenant le data generator (ex: full_train_datagen.pkl)"
    )
    args = parser.parse_args()

    # Charge le data generator depuis le pickle fourni en argument
    dataset_pickle_path = args.dataset_pickle
    with open(dataset_pickle_path, "rb") as f:
        full_train_datagen = pickle.load(f)

    results = []
    for i in range(full_train_datagen.num_samples):
        results.append(count_classes_for_index(i, full_train_datagen))

    counter = Counter()
    for d in results:
        counter.update(d)

    # Création du dossier si besoin
    output_dir = "count_classes"
    os.makedirs(output_dir, exist_ok=True)

    # Génération du nom de fichier avec date, heure et nom du pickle
    base_name = os.path.splitext(os.path.basename(dataset_pickle_path))[0]
    output_path = os.path.join(output_dir, f"class_counter_{base_name}.pkl")

    # Sauvegarde le résultat
    with open(output_path, "wb") as f:
        pickle.dump(counter, f)
    print(f"Résultat sauvegardé dans {output_path}")