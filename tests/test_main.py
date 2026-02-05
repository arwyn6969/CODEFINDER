"""
Basic tests for the main application
"""
import pytest
from fastapi.testclient import TestClient
from app.api.main import app


# Lazy client initialization to avoid module-level instantiation errors
_client = None


def get_client():
    """Get or create the test client."""
    global _client
    if _client is None:
        _client = TestClient(app)
    return _client


# Create a proxy object that lazily accesses the client
class LazyClient:
    def __getattr__(self, name):
        return getattr(get_client(), name)


client = LazyClient()


def test_root_endpoint():
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200


def test_health_check():
    """Test the health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "version" in response.json()


def test_upload_endpoint_no_file():
    """Test upload endpoint without file"""
    response = client.post("/api/documents/upload")
    assert response.status_code == 422  # Validation error


def test_api_routes():
    """Test basic API routes"""
    response = client.get("/api/documents/")
    assert response.status_code == 200