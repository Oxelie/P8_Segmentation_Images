

# Ce script Streamlit permet de sélectionner une image, d’afficher l’image et les masques, et d’appeler l’API.

import streamlit as st
import numpy as np
from PIL import Image
import pathlib
import requests
from api_client import predict_mask

# Chemin vers le dossier test
TEST_IMAGE_DIR = pathlib.Path("data/test")
API_URL = "https://segmentation-api-std-8768959d234a.herokuapp.com/"

# Récupère la liste des images et masques
image_paths = sorted(list(TEST_IMAGE_DIR.glob("*leftImg8bit.png")))
mask_paths = sorted(list(TEST_IMAGE_DIR.glob("*labelIds.png")))
test_samples = list(zip(image_paths, mask_paths))

st.title("Segmentation d'images - Demo")

# Sélection de l'image
image_ids = [img.name for img, _ in test_samples]
selected_id = st.selectbox("Sélectionne une image du jeu de test :", image_ids)

# Récupère les chemins
selected_img_path, selected_mask_path = next((img, mask) for img, mask in test_samples if img.name == selected_id)

# Affiche l'image originale
orig_img = Image.open(selected_img_path)
st.image(orig_img, caption="Image originale", use_column_width=True)

# Affiche le masque réel
real_mask = np.array(Image.open(selected_mask_path))
st.image(real_mask, caption="Masque réel", use_column_width=True)

# Prédiction via l'API
with open(selected_img_path, "rb") as f:
    image_bytes = f.read()
if st.button("Lancer la prédiction du masque"):
    try:
        mask_pred = predict_mask(API_URL, image_bytes, model_name="unet_mini")
        mask_pred = np.array(mask_pred)
        st.image(mask_pred, caption="Masque prédit", use_column_width=True)
    except Exception as e:
        st.error(f"Erreur lors de la prédiction : {e}")