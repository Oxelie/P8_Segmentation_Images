import pytest
from api import app

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