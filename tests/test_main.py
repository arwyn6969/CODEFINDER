"""
Basic tests for the main application
"""
import pytest
from fastapi.testclient import TestClient
from app.api.main import app

client = TestClient(app)

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