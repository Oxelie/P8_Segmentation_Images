from flask import Flask, jsonify, request
import os
import numpy as np
from tensorflow import keras
import io
from PIL import Image   
import tensorflow as tf
from keras.applications.mobilenet_v3 import preprocess_input as mnv3_preprocess_input
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess_input
from .custom_object import DiceFocalLoss
from .classe_dataset import ImageSegmentationDataset
import pathlib
from .utils_p8 import labels
import matplotlib.pyplot as plt
import base64



# Charge le modèle entraîné
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model_ResNet50_UNet.keras") 
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
test_dir = pathlib.Path("../test")


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
    model_name="resnet50_unet") #pour normalisation adpatée au modèle



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
    
    # on retourne la liste des images de test disponibles (sans les chemins complets, juste les noms d'images) et leurs index 
    return jsonify([{"index": i, "path": str(p).replace("../test/", "")} for i, p in enumerate(test_dataset.image_paths)])



# sélectionner une image de test parmi la liste des images de test disponibles (endpoint /list_img)
@app.route("/select_img", methods=["POST"])
def select_image():
   
    # # récupérer en plus le mask vérité terrain _gtFine_color.png  
    # test_dataset.image_paths[request.json["image_index"]] # chemin de l'image sélectionnée
    # test_dataset.mask_paths[request.json["image_index"]] # chemin du masque de vérité terrain correspondant à l'image sélectionnée
    
     # pour charger l'image et le masque de vérité terrain sélectionnés dans le dataset (pour affichage dans l'app streamlit)
    
    _, mask, paths = test_dataset.get_image_and_mask(request.json["image_index"])
    orig_img = np.array(Image.open(test_dataset.image_paths[request.json["image_index"]]).convert("RGB").resize((TARGET_SIZE[1], TARGET_SIZE[0]), Image.BILINEAR)
)

    
    num_classes= 8
    cmap = plt.get_cmap("tab10", num_classes)

    fig, axs = plt.subplots(1, 3, figsize=(15, 6))

    axs[0].imshow(orig_img)
    axs[0].set_title("Original Image")
    axs[1].imshow(mask, cmap=cmap, vmin=0, vmax=num_classes-1)
    axs[1].set_title("Ground Truth Mask")
    
    for ax in axs:
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    

    bbox_orig = axs[0].get_tightbbox(fig.canvas.get_renderer())
    bbox_orig = bbox_orig.transformed(fig.dpi_scale_trans.inverted())
    
    bbox_mask = axs[1].get_tightbbox(fig.canvas.get_renderer())
    bbox_mask = bbox_mask.transformed(fig.dpi_scale_trans.inverted())
    
    
    buf = io.BytesIO()
    fig.savefig(buf, format='png', 
                bbox_inches=bbox_orig
                )   
    buf.seek(0)
    
    buf_mask = io.BytesIO()
    fig.savefig(buf_mask, format='png', 
                bbox_inches=bbox_mask
                )   
    buf_mask.seek(0)
    
    encoded_orig = base64.b64encode(buf.read()).decode('utf-8')
    encoded_mask = base64.b64encode(buf_mask.read()).decode('utf-8')

    return jsonify({
        "image": encoded_orig,
        "mask": encoded_mask
    })  
    

# faire un 2eme API pour la prédiction du masque (endpoint /predict) 
@app.route("/predict", methods=["POST"])
def predict_mask():    
    
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