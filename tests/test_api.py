import pytest
from api import app


import requests
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from api import app

def test_predict_endpoint():
    url = "http://localhost:5000/predict"
    # Utilise une petite image de test
    with open("tests/test_image.png", "rb") as img:
        files = {"image": img}
        response = requests.post(url, files=files)
    assert response.status_code == 200
    data = response.json()
    assert "mask" in data
    

@pytest.fixture
def client():
    with app.test_client() as client:
        yield client

def test_predict_endpoint(client):
    # Simule l'envoi d'une image à l'API
    with open("tests/test_image.png", "rb") as img:
        response = client.post("/predict", data={"image": img})
    assert response.status_code == 200
    json_data = response.get_json()
    assert "mask" in json_data