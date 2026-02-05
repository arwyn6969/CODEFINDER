"""
Basic tests for the main application
"""
import pytest
from fastapi.testclient import TestClient
from app.api.main import app


@pytest.fixture
def client():
    """Create a test client for each test."""
    with TestClient(app) as c:
        yield c

def test_root_endpoint(client):
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200

def test_health_check(client):
    """Test the health check endpoint"""
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert "version" in response.json()

def test_upload_endpoint_no_file(client):
    """Test upload endpoint without file"""
    response = client.post("/api/documents/upload")
    assert response.status_code == 422  # Validation error

def test_api_routes(client):
    """Test basic API routes"""
    response = client.get("/api/documents/")
    assert response.status_code == 200