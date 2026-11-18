from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status":"ok"}

def test_predict_positive():
    response = client.post("/predict", json={"text":"An amazing, wonderful film. I loved it!"})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "positive"

def test_predict_negative():
    response = client.post("/predict", json={"text":"Terrible, boring movie. A total waste of time."})
    assert response.status_code == 200
    assert response.json()["sentiment"] == "negative"

def test_empty_text_rejected():
    response = client.post("/predict", json={"text":"  "})
    assert response.status_code == 422

def test_missing_text_rejected():
    response = client.post("/predict", json={})
    assert response.status_code == 422