"""
Tests for the FastAPI serving application.

Tests API endpoints, request/response handling, and error cases.
"""

import pytest
from fastapi.testclient import TestClient
from pathlib import Path
import sys

# Import the FastAPI app
sys.path.insert(0, str(Path(__file__).parent.parent))

# These tests will only run if a trained model exists
pytestmark = pytest.mark.skipif(
    not (Path(__file__).parent.parent / "best_model").exists(),
    reason="No trained model found (best_model/ directory missing)",
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    from serve import app

    return TestClient(app)


def test_health_endpoint(client):
    """Test the /health endpoint."""
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()

    assert "status" in data
    assert data["status"] == "healthy"
    assert "model_loaded" in data
    assert data["model_loaded"] is True


def test_predict_endpoint_single(client, sample_texts):
    """Test the /predict endpoint with a single text."""
    # Test positive sentiment
    payload = {"text": sample_texts["positive"][0]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "text" in data
    assert "label" in data
    assert "confidence" in data
    assert "probabilities" in data
    assert "prediction_time_ms" in data

    assert data["text"] == sample_texts["positive"][0]
    assert isinstance(data["label"], int)
    assert isinstance(data["confidence"], float)
    assert 0 <= data["confidence"] <= 1
    assert isinstance(data["probabilities"], list)
    assert isinstance(data["prediction_time_ms"], (int, float))


def test_predict_endpoint_negative_sentiment(client, sample_texts):
    """Test prediction on negative sentiment text."""
    payload = {"text": sample_texts["negative"][0]}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "label" in data


def test_predict_endpoint_empty_text(client):
    """Test /predict endpoint with empty text."""
    payload = {"text": ""}
    response = client.post("/predict", json=payload)

    # Should still return a prediction (model handles empty text)
    assert response.status_code == 200
    data = response.json()
    assert "label" in data


def test_predict_endpoint_missing_text(client):
    """Test /predict endpoint with missing text field."""
    payload = {}
    response = client.post("/predict", json=payload)

    # Should return validation error (422 Unprocessable Entity)
    assert response.status_code == 422


def test_predict_endpoint_invalid_type(client):
    """Test /predict endpoint with invalid text type."""
    payload = {"text": 12345}  # Integer instead of string
    response = client.post("/predict", json=payload)

    # Should return validation error
    assert response.status_code == 422


def test_batch_predict_endpoint(client, sample_texts):
    """Test the /batch_predict endpoint."""
    texts = sample_texts["positive"][:2] + sample_texts["negative"][:2]
    payload = {"texts": texts}
    response = client.post("/batch_predict", json=payload)

    assert response.status_code == 200
    data = response.json()

    assert "predictions" in data
    assert "total_prediction_time_ms" in data

    predictions = data["predictions"]
    assert len(predictions) == len(texts)

    for i, pred in enumerate(predictions):
        assert "text" in pred
        assert "label" in pred
        assert "confidence" in pred
        assert pred["text"] == texts[i]


def test_batch_predict_empty_list(client):
    """Test /batch_predict with empty list."""
    payload = {"texts": []}
    response = client.post("/batch_predict", json=payload)

    # Should return 400 for empty list
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_batch_predict_too_many_texts(client):
    """Test /batch_predict with too many texts."""
    # Create more than max allowed (assuming max is 100)
    texts = ["Test text"] * 150
    payload = {"texts": texts}
    response = client.post("/batch_predict", json=payload)

    # Should return 400 for too many texts
    assert response.status_code == 400


def test_batch_predict_missing_texts(client):
    """Test /batch_predict with missing texts field."""
    payload = {}
    response = client.post("/batch_predict", json=payload)

    # Should return validation error
    assert response.status_code == 422


def test_root_endpoint(client):
    """Test the root endpoint."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()

    assert "message" in data
    assert "model" in data
    assert "endpoints" in data


def test_docs_endpoint(client):
    """Test that /docs endpoint is accessible."""
    response = client.get("/docs")

    # Should return HTML page
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_openapi_schema(client):
    """Test that OpenAPI schema is accessible."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    data = response.json()

    assert "openapi" in data
    assert "info" in data
    assert "paths" in data

    # Verify our endpoints are documented
    assert "/health" in data["paths"]
    assert "/predict" in data["paths"]
    assert "/batch_predict" in data["paths"]


def test_concurrent_requests(client, sample_texts):
    """Test handling multiple concurrent requests."""
    import concurrent.futures

    def make_request():
        payload = {"text": sample_texts["positive"][0]}
        return client.post("/predict", json=payload)

    # Make 10 concurrent requests
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(10)]
        responses = [f.result() for f in concurrent.futures.as_completed(futures)]

    # All should succeed
    assert all(r.status_code == 200 for r in responses)
