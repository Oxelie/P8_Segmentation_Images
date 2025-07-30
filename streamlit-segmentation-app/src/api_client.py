

# Ce script gère la communication avec l’API Flask.

import requests

def predict_mask(api_url, image_bytes, model_name="unet_mini"):
    """
    Envoie une image à l'API Flask et récupère le masque prédit.
    """
    files = {"image": image_bytes}
    data = {"model_name": model_name}
    response = requests.post(f"{api_url}/predict", files=files, data=data)
    response.raise_for_status()
    return response.json()["mask"]