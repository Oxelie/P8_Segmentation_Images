from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Charge le modèle entraîné
MODEL_PATH = os.path.join(os.path.dirname(__file__), "../model_tf")
model = tf.keras.models.load_model(MODEL_PATH)

# Taille attendue par le modèle
TARGET_SIZE = (256, 512)

# def preprocess_image(image_bytes):
#     img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
#     img = img.resize(TARGET_SIZE, Image.BILINEAR)
#     img_arr = np.array(img, dtype=np.float32) / 255.0
#     img_arr = np.expand_dims(img_arr, axis=0)
#     return img_arr


# @app.route("/predict", methods=["POST"])
# def predict():
#     if "image" not in request.files:
#         return jsonify({"error": "No image provided"}), 400
#     file = request.files["image"]
#     img_bytes = file.read()
#     img_arr = preprocess_image(img_bytes)
#     pred = model.predict(img_arr)
#     mask = np.argmax(pred, axis=-1)[0]
#     # Pour simplifier, on retourne le mask sous forme de liste (à adapter selon besoin)
#     return jsonify({"mask": mask.tolist()})

def preprocess_image(image_bytes, model_name="unet_mini"):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32)
    if model_name.lower() in ["unet", "unet_mini"]: #UNet et UNet_mini : normalisation entre 0 et 1
        img_arr = img_arr / 255.0
    elif model_name.lower() == "vgg16_unet": #VGG16_UNet : normalisation entre -1 et 1
        img_arr = (img_arr / 127.5) - 1.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    img_bytes = file.read()
    # Récupère le nom du modèle depuis la requête ou utilise la valeur par défaut
    model_name = request.form.get("model_name", "unet_mini")
    img_arr = preprocess_image(img_bytes, model_name=model_name)
    pred = model.predict(img_arr)
    mask_pred = np.argmax(pred, axis=-1)[0]
    return jsonify({"mask": mask_pred.tolist()})


if __name__ == "__main__":
    app.run(debug=True)