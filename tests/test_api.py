# API tests: do the API's "doors" work?
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

GOOD_CAR = {"year": 2017, "cc": 1300, "mileage": 55000, "imported": 0, "automatic": 0}

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_returns_a_price():
    response = client.post("/predict", json=GOOD_CAR)
    assert response.status_code == 200
    assert response.json()["price_lakh"] > 0

def test_bad_input_is_rejected():
    bad = {**GOOD_CAR, "year": "abc"}
    assert client.post("/predict", json=bad).status_code == 422

def test_out_of_range_flag_is_rejected():
    bad = {**GOOD_CAR, "imported": 5}
    assert client.post("/predict", json=bad).status_code == 422