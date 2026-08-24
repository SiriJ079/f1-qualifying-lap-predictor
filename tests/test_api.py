from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_metadata_endpoint():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "drivers" in data
    assert "circuits" in data

def test_predict_valid_request():
    metadata = client.get("/metadata").json()
    driver = metadata["drivers"][0]
    circuit = metadata["circuits"][0]

    response = client.post("/predict", json={"driver": driver, "circuit": circuit, "year": 2026})
    assert response.status_code == 200
    body = response.json()
    assert body["driver"] == driver

def test_predict_unknown_driver_returns_404():
    response = client.post("/predict", json={"driver": "ZZZ", "circuit": "Italian Grand Prix", "year": 2026})
    assert response.status_code == 404

def test_compare_multiple_drivers():
    metadata = client.get("/metadata").json()
    drivers = metadata["drivers"][:3]
    circuit = metadata["circuits"][0]

    response = client.post("/compare", json={"drivers": drivers, "circuit": circuit, "year": 2026})
    assert response.status_code == 200
    assert len(response.json()["predictions"]) > 0

def test_homepage_renders():
    response = client.get("/")
    assert response.status_code == 200
    assert "F1 Qualifying Predictor" in response.text