"""
Tests for the API
"""

import os

from fastapi.testclient import TestClient

os.environ["TESTING"] = "true"

from src.app import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "MLOps Demo API"
    assert data["status"] == "running"


def test_health_endpoint():
    """Test health check"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data
    assert "version" in data


def test_predict_endpoint():
    """Test prediction endpoint"""
    payload = {"features": [1.0, 2.0, 3.0, 4.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "confidence" in data
    assert "model_version" in data
    assert isinstance(data["prediction"], float)
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_invalid_features():
    """Test with invalid features"""
    # Not enough features
    payload = {"features": [1.0]}
    response = client.post("/predict", json=payload)
    assert response.status_code == 400

    # Empty features
    payload = {"features": []}
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Validation error


def test_model_info_endpoint():
    """Test model info endpoint"""
    response = client.get("/model/info")
    assert response.status_code == 200
    data = response.json()
    assert "model_type" in data
    assert "version" in data
    assert "features_expected" in data
    assert data["status"] == "running"
