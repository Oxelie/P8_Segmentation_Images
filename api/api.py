from flask import Flask, jsonify, request
import os
import numpy as np
from tensorflow import keras
import io
from PIL import Image   
import tensorflow as tf
from keras.applications.mobilenet_v3 import preprocess_input as mnv3_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from custom_object import DiceFocalLoss
from classe_dataset import ImageSegmentationDataset
import pathlib
from utils_p8 import labels



# Charge le modèle entraîné
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../models/models/ResNet50_UNet/model_ResNet50_UNet.keras") 
# actuellement dans le dossier /Users/stephanieduhem/Documents/_DIPLOMES_CURSUS_/MASTER_AI_ENGINEER/openclassroom/projet_8/P8_Segmentation_Images/mlf_1/models 
# + sous dossiers aux noms des modèles REsNet50_Unet_50epochs_data_augm


# définir le nom du modèle - en récupérant le nom du modèle à partir du nom du fichier du modèle chargé
model_name = os.path.basename(MODEL_PATH).split(".")[0].lower() 

model = tf.keras.models.load_model(
    MODEL_PATH,
    custom_objects={"DiceFocalLoss": DiceFocalLoss},
    compile=False
    )

# Taille attendue par le modèle
TARGET_SIZE = (256, 512)


# path vers les nouveaux dossier train et test
test_dir = pathlib.Path("../data/test")


# récupération des chemins pour les images et les masques du test
image_paths_test = sorted(list(test_dir.glob("*leftImg8bit.png")))
mask_paths_test = sorted(list(test_dir.glob("*labelIds.png")))
test_paths = list(zip(image_paths_test, mask_paths_test))

test_dataset = ImageSegmentationDataset(
    paths=test_paths,
    labels=labels,
    batch_size=4,
    augmentations=False,        # Pas d'augmentation pour le test
    normalize=True,
    shuffle=False,
    label_onehot=False,
    sample_weights=True,        # Poids de classes activés
    model_name="unet_mini") #pour normalisation adpatée au modèle



# fonction de prétraitement des images d'entrée, adaptée selon le modèle choisi
def preprocess_img(img_array: np.ndarray) -> np.ndarray:
        """
        Applique la normalisation des images adaptée selon le modèle choisi.
        - UNet et UNet_mini : normalisation entre 0 et 1
        - VGG16_UNet : normalisation entre -1 et 1
        - MobileNetV3_UNet : tf.keras.applications.mobilenet_v3.preprocess_input ([-1, 1])
        - ResNet50_UNet : resnet_preprocess_input (RGB->BGR + mean subtraction, images en [0,255])
        """
        if not isinstance(img_array, np.ndarray):
            img_array = np.array(img_array)

        # si image encodée en [0,1], remultiplie pour revenir à [0,255] avant preprocess_input
        needs_rescale = img_array.dtype != np.uint8 and img_array.max() <= 1.0
    
        if isinstance(self.normalize, bool) and self.normalize:
            if self.model_name in ["unet", "unet_mini"]:
                return img_array / 255.0
            elif self.model_name == "vgg16_unet": 
                return (img_array / 127.5) - 1.0
            elif self.model_name == "mobilenetv3small_unet":
                return mnv3_preprocess_input(img_array)
            elif self.model_name in ["resnet50_unet", "resnet_unet"]:
                arr = img_array.astype(np.float32)
                if needs_rescale:
                    arr = arr * 255.0
            # resnet_preprocess_input gère conversion RGB->BGR et soustraction des moyennes
            return resnet_preprocess_input(arr)
        return img_array



app = Flask(__name__)

# cet API c'est pour la liste des images de test 
@app.route("/list_img", methods=["GET"])
def list_images():
    # filenames = test_dataset.image_paths
    # usable_img = []
    # for filename in filenames:
    #     if filename.endswith("_leftImg8bit.png") :
    #         usable_img.append(filename)
    # print("test_dataset.image_paths:", test_dataset.image_paths)
    return jsonify([str(p) for p in test_dataset.image_paths])

selected_img = None

# sélectionner une image de test parmi la liste des images de test disponibles (endpoint /list_img)
@app.route("/select_img", methods=["POST"])
def select_image():
    global selected_img
    selected_img = request.json["image_name"]
    
    # récupérer en plus le mask vérité terrain _gtFine_color.png  
    
    return jsonify({"selected_image": selected_img})

# recupérer l'index de l'image sélectionnée coté front 

# 





# faire un 2eme API pour la prédiction du masque (endpoint /predict) 
@app.route("/predict", methods=["POST"])
def predict_mask():
    global test_dataset
    
    
    img_mask_pred = test_dataset.show_prediction(model, request.json["image_index"])
    
    return jsonify({"img_mask_pred": img_mask_pred})
    
    # global selected_img
    # # ici on va recevoir le nom de l'image de test à prédire
    # # selected_img = jsonify(usable_img)[0] 
    # # charger l'image de test correspondante depuis le dossier data/test
    # img_path = os.path.join("../data/test", selected_img)
    # open_img = Image.open(img_path).convert("RGB")
    # # preprocess de l'image pour la mettre au format attendu par le modèle (taille, normalisation)
    # resized_img = open_img.resize(TARGET_SIZE)
    # prep_img = preprocess_img(resized_img)    
    # # ensuite on va faire la prédiction du masque pour cette image
    # # et enfin on va retourner le masque prédit (sous forme d'image ou de tableau numpy)
    # mask_pred = model.predict(np.expand_dims(prep_img, axis=0))
    # mask_pred = np.argmax(mask_pred.squeeze(), axis=-1)
    # return mask_pred 

# sachant que  l'affichage des images de test + image du mask réel + image du mask prédit est dans une autre app (streamlit)
# on va créer une classe à part pour l'affichage des images et des masques dans l'app


if __name__ == "__main__":
    app.run(port = 4444, debug=True)