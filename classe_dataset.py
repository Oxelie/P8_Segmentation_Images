
# data & science
import math
import numpy as np
from collections import Counter

# system & tools
import pathlib

# POO
from functools import cached_property
from typing import Optional, Union, Tuple, List, NamedTuple, Any

# graphiques
import matplotlib.pyplot as plt
from matplotlib.image import imread

# computer vision / CNN
from PIL import Image
import albumentations as A
import tensorflow as tf

# TensorFlow utilise un seul thread pour garantir la reproductibilité
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# Import local
from utils_p8 import labels, TARGET_SIZE, COL_ORDER, show_nums_axes


# Création d'une classe pour la création d'un dataset de segmentation d'image
# Hérite de tf.keras.utils.PyDataset pour bénéficier des fonctionnalités de dataset de TensorFlow
# Permet de charger, transformer et prétraiter des images et des masques pour l'entraînement d'un modèle de segmentation d'image.
class ImageSegmentationDataset(tf.keras.utils.PyDataset):
    """
    Générer un dataset adapté à la segmentation d'image.
    """
    # Dimension attendue 
    TARGET_SIZE = (256, 512)

    # Initialisation de l'objet dataset
    def __init__( # déclaration du constructeur
        self, # référence à l'instance de la classe
        paths: List[Tuple[pathlib.Path, pathlib.Path]], # liste de tuples contenant les chemins des images et des masques
        labels: List[NamedTuple], # liste des labels pour la correspondance des catégories
        batch_size: int, # taille des batchs d'images pour chaque itération de l'entraînement
        augmentations: bool = False, # booléen pour l'augmentation des données
        preview: Optional[int] = None, # nombre d'échantillons à prévisualiser (None pour tout charger)
        normalize: Union[bool, str] = True, # booléen pour normalisation des données
        shuffle: bool = True, # booléen pour mélanger les données à chaque époque
        label_onehot: bool = False, # booléen pour la conversion en one_hot encoding des classes
        sample_weights: Optional[List[float]] = None, # poids pour la pondération des classes en cas de déséquilibre
        model_name: Optional[str] = "unet", # nom du modèle utilisé pour la segmentation, par défaut "unet"
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs) # appelle le constructeur de la classe parente

        # load les chemins des images et des masques et applique un aperçu si nécessaire
        self.image_paths, self.mask_paths = self.load_img_and_mask_paths(paths, preview)

        # table de correspondances entre les IDs des labels et les catégorie
        self.table_id2category = {label.id: label.categoryId for label in labels}
        self.table_category2name = {label.categoryId: label.category for label in labels}

        # init les paramètres du dataset
        if not self.image_paths or not self.mask_paths:
            raise ValueError("Image and mask paths cannot be empty.")
        self.batch_size = batch_size
        self.augmentations = augmentations
        self.normalize = normalize
        self.shuffle = shuffle
        self.label_onehot = label_onehot
        self.sample_weights = sample_weights
        self.model_name = model_name.lower() if model_name else "unet"

        # Pipeline d'augmentations si requise
        if self.augmentations:
            # créationd d'un objet 'Compose' issu d’Albumentations pour enchaîner plusieurs transformations.
            self.compose = A.Compose(
                [
                    A.HorizontalFlip(p=0.5), # retournement horizontal avec une probabilité de 50 %
                    #  soit une variation aléatoire de la luminosité/contraste, soit une variation de la teinte/saturation/valeur, avec une probabilité totale de 50 %.
                    # Chaque transformation interne a une probabilité de 100 % d’être appliquée si choisie.
                    A.OneOf(
                        [
                            A.RandomBrightnessContrast(
                                brightness_limit=0.2, contrast_limit=0.2, p=1.0
                            ),
                            A.HueSaturationValue(
                                hue_shift_limit=10,
                                sat_shift_limit=15,
                                val_shift_limit=10,
                                p=1.0,
                            ),
                        ],
                        p=0.5,
                    ),
                    # soit un flou gaussien, soit un flou de mouvement, soit une distorsion optique, avec une probabilité totale de 25 %.
                    # Chaque transformation interne a une probabilité de 100 % d’être appliquée si choisie.
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=3, p=1.0),
                            A.MotionBlur(blur_limit=5, p=1.0),
                            A.OpticalDistortion(distort_limit=0.05, p=1.0),
                        ],
                        p=0.25,
                    ),
                ],
                # si des poids sont utilisés, on indique à Albumentations de traiter sample_weights comme un masque (pour appliquer les mêmes transformations que sur le masque principal).
                additional_targets={"sample_weights": "mask"}
                if self.sample_weights is not None
                else {},
            )

        # mélange les paths des images et des masques si demandé à chaque époch, pour améliorer la généralisation du modèle.
        if self.shuffle:
            self.on_epoch_end()
            
    # intérêt des décorateurs (= @cached_property) : transformer une méthode en propriété calculée une seule fois et mise en cache.
    @cached_property 
    def num_classes(self) -> int:
        """retourne le nombre de classes différentes dans le dataset"""
        return len(set(self.table_id2category.values()))

    @cached_property 
    def num_samples(self) -> int:
        """ retourne le nombre total d’images dans le dataset"""
        return len(self.image_paths)
    
    @cached_property
    def class_pixel_counts(self) -> dict:
        """
        Compte le nombre de pixels par classe sur l'ensemble des masques du dataset.
        Retourne un dictionnaire {classe: nombre de pixels}.
        """
        counter = Counter()
        for mask_path in self.mask_paths:
            mask = self.load_mask_to_array(mask_path)
            unique, counts = np.unique(mask, return_counts=True)
            counter.update(dict(zip(unique, counts)))
        return dict(counter)


    # @staticmethod = méthode utilitaire liée à la classe, mais indépendante de l’état de l’objet.
    # Cette méthode n’utilise pas l’instance (self) ni la classe elle-même (cls), elle agit uniquement sur les arguments explicitement passés à la fonction.
    @staticmethod  
    def load_img_and_mask_paths(
        paths: List[Tuple[pathlib.Path, pathlib.Path]], preview: Optional[int]
    ) -> Tuple[List[pathlib.Path], List[pathlib.Path]]:
        """Unpack les tuples des chemins des images et des masques correspondant et applique une sélection sur les premiers éléments si l'on souhaite une préview"""
        image_paths, mask_paths = zip(*paths)
        if len(image_paths) != len(mask_paths):
            raise ValueError("Number of images and masks must be equal.")
        if preview is not None:
            image_paths = image_paths[:preview]
            mask_paths = mask_paths[:preview]
        return list(image_paths), list(mask_paths)

    def __len__(self) -> int:
        """avec math.ceil s'assurer que tous les samples sont pris, même si le dernier batch est plus petit que les autres 
        et retourne le nb de lots (batchs) par epoch d'entraînement en fonction de la taille de lot spécifiée (batchsize)."""
        length = math.ceil(self.num_samples / self.batch_size)
        return length

    def __getitem__(
        self, index: int
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """Récupère un batch d’images et de masques à l’index donné (pour l’entraînement ou la validation)
        calcul des index de début et de fin de chaque batch, récupère les paths des images et des masques correspondants,
        charge et trasnforme les images et les masques via 'load_and_augment' 
        et retourne les images, masques et les poids d'échantillons s'ils sont fournis dans le cas de classes déséquilibrées."""
        start_idx = index * self.batch_size
        end_idx = min(start_idx + self.batch_size, self.num_samples)
        if start_idx >= self.num_samples:
            raise IndexError("Index out of range")

        batch_paths = list(
            zip(self.image_paths[start_idx:end_idx], self.mask_paths[start_idx:end_idx])
        )
        results = [self.load_and_augment(pair) for pair in batch_paths]

        if self.sample_weights is not None:
            images, masks, weights = zip(*results)
            return np.asarray(images), np.asarray(masks), np.asarray(weights)
        else:
            images, masks = zip(*results)
            return np.asarray(images), np.asarray(masks)

    def on_epoch_end(self) -> None:
        """Mélange aléatoirement les couples (image, masque) à la fin de chaque époch si option activée, 
        pour améliorer la généralisation du modèle."""
        if self.shuffle:
            zip_paths = list(zip(self.image_paths, self.mask_paths))
            np.random.shuffle(zip_paths)
            self.image_paths, self.mask_paths = zip(*zip_paths)

    def preprocess_img(self, img_array: np.ndarray) -> np.ndarray:
        """
        Applique la normalisation des images adaptée selon le modèle choisi.
        - UNet et UNet_mini : normalisation entre 0 et 1
        - VGG16_UNet : normalisation entre -1 et 1
        """
        if isinstance(self.normalize, bool) and self.normalize:
            if self.model_name in ["unet", "unet_mini"]:
                return img_array / 255.0
            elif self.model_name == "vgg16_unet": 
                return (img_array / 127.5) - 1.0
        return img_array

    def load_img_to_array(self, img_path: pathlib.Path) -> np.ndarray:
        """Charge, resize et convertit en tableau numpy (float32) une image,
        puis applique la normalisation avec 'preprocess_img'"""
        img = tf.keras.utils.load_img(
            str(img_path),
            target_size=self.TARGET_SIZE,
            color_mode="rgb",
            interpolation="bilinear",
        )
        img_array = tf.keras.utils.img_to_array(img, dtype=np.float32)
        return self.preprocess_img(img_array)

    def load_mask_to_array(self, mask_path: pathlib.Path) -> np.ndarray:
        """load en niveau de gris, resize un masque,
        remplace pour chaque pixel les IDs des labels par les IDs des catégories mères correspondantes,
        si spécifié, convertit le masque en one-hot encoding."""
        mask = tf.keras.utils.load_img(
            str(mask_path),
            target_size=self.TARGET_SIZE,
            color_mode="grayscale",
            interpolation="nearest",
        )
        mask_array = tf.keras.utils.img_to_array(mask, dtype=np.int8)
        mask_array = np.vectorize(self.table_id2category.get)(mask_array).squeeze()
        if self.label_onehot:
            mask_array = tf.keras.utils.to_categorical(
                mask_array, num_classes=self.num_classes
            )
        return mask_array

    def load_and_augment(
        self, paths: Tuple[pathlib.Path, pathlib.Path]
    ) -> Union[
        Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]
    ]:
        """charge l’image et le masque correspondant, applique sur le masque les poids pour les classes déséquilibrées si spécifié,
        si les augmentations sont activées, applique les transformations avec Albumentations sur l'image et le masque (et les poids si présents)
        """
        img_path, mask_path = paths
        img = self.load_img_to_array(img_path)
        mask = self.load_mask_to_array(mask_path)

        if self.sample_weights is not None:
            weights = np.take(self.sample_weights, mask)
            if self.augmentations:
                augmented = self.compose(image=img, mask=mask, sample_weights=mask)
                return (
                    augmented["image"],
                    augmented["mask"],
                    augmented["sample_weights"],
                )
            else:
                return img, mask, weights
        else:
            if self.augmentations:
                augmented = self.compose(image=img, mask=mask)
                return augmented["image"], augmented["mask"]
            else:
                return img, mask
            
    def get_image_and_mask(
        self, index: int
    ) -> Tuple[np.ndarray, np.ndarray, Tuple[pathlib.Path, pathlib.Path]]:
        """ retourne l'image, le masque et les paths correspondants pour un index donné (pour visualisation ou prédiction)"""
        paths = (self.image_paths[index], self.mask_paths[index])
        if self.sample_weights is None:
            img, mask = self.load_and_augment(paths)
        else:
            img, mask, _ = self.load_and_augment(paths)
        return img, mask, paths

    def get_prediction(self, model: Any, index: int) -> np.ndarray:
        """utilise le modèle choisi pour prédire le masque d'une image à l'index donné"""
        img, _, _ = self.get_image_and_mask(index)
        mask_pred = model.predict(np.expand_dims(img, axis=0))
        mask_pred = np.argmax(mask_pred.squeeze(), axis=-1)
        return mask_pred

    def show_transformation(
        self, index: int, figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """ affiche l'image et le masque d'origine et ceux transformés pour un échantillon donné"""
        img, mask, paths = self.get_image_and_mask(index)
        img_path, mask_path = paths

        orig_img = np.array(Image.open(img_path))
        orig_mask = np.array(Image.open(mask_path))

        if self.label_onehot:
            mask = np.argmax(mask, axis=-1)
        if isinstance(self.normalize, str):
            img = (img - img.min()) / (img.max() - img.min())

        fig, ax = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle("Image et masque - avant/après transformations", fontsize=16)
        ax[0, 0].imshow(orig_img)
        ax[0, 0].set_title("Original Image")
        ax[0, 1].imshow(img.astype(np.uint8) if img.max() > 1 else img)
        ax[0, 1].set_title("Transformed Image")
        ax[1, 0].imshow(orig_mask)
        ax[1, 0].set_title("Original Mask")
        ax[1, 1].imshow(mask)
        ax[1, 1].set_title("Transformed Mask")
        for a in ax.ravel():
            a.axis("off")
        plt.tight_layout()
        plt.show()

    def show_prediction(
        self, model: Any, index: int, figsize: Tuple[int, int] = (15, 6)
    ) -> None:
        """affiche l'image originale, le masque d'origine (réalité terrain) et le masque prédit du modèle pour un échantillon"""
        img, mask, paths = self.get_image_and_mask(index)
        img_path, mask_path = paths
        mask_pred = self.get_prediction(model, index)

        orig_img = np.array(Image.open(img_path))

        if self.label_onehot:
            mask = np.argmax(mask, axis=-1)
        if isinstance(self.normalize, str):
            img = (img - img.min()) / (img.max() - img.min())

        fig, axs = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle(f"{model.name} Predictions", fontsize=16)

        axs[0].imshow(orig_img)
        axs[0].set_title("Original Image")
        axs[1].imshow(mask, cmap="Greys")
        axs[1].set_title("Ground Truth Mask")
        axs[2].imshow(mask_pred, cmap="Greys")
        axs[2].set_title("Predicted Mask")
        for ax in axs:
            ax.axis("off")
        plt.tight_layout()
        plt.show()