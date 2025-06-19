import numpy as np
import pathlib
import pickle
from collections import Counter
import multiprocessing as mp
import os
from datetime import datetime
import argparse
from classe_dataset import ImageSegmentationDataset

def count_classes_for_index(i):
    _, mask, _ = full_train_datagen.get_image_and_mask(i)
    unique, counts = np.unique(mask, return_counts=True)
    return dict(zip(unique, counts))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compte les classes d'un dataset de segmentation.")
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

    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(count_classes_for_index, range(full_train_datagen.num_samples))

    counter = Counter()
    for d in results:
        counter.update(d)

    # Création du dossier si besoin
    output_dir = "count_classes"
    os.makedirs(output_dir, exist_ok=True)

    # Génération du nom de fichier avec date, heure et nom du pickle
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(dataset_pickle_path))[0]
    output_path = os.path.join(output_dir, f"class_counter_{base_name}_{now}.pkl")

    # Sauvegarde le résultat
    with open(output_path, "wb") as f:
        pickle.dump(counter, f)