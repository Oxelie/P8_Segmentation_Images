from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Charge le modèle entraîné (dossier SavedModel ou .h5)
model = tf.keras.models.load_model("model_tf")

# Taille attendue par le modèle
TARGET_SIZE = (256, 512)

def preprocess_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.BILINEAR)
    img_arr = np.array(img, dtype=np.float32) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image provided"}), 400
    file = request.files["image"]
    img_bytes = file.read()
    img_arr = preprocess_image(img_bytes)
    pred = model.predict(img_arr)
    mask = np.argmax(pred, axis=-1)[0]
    # Pour simplifier, on retourne le mask sous forme de liste (à adapter selon besoin)
    return jsonify({"mask": mask.tolist()})

if __name__ == "__main__":
    app.run(debug=True)