"""
Pytest configuration and shared fixtures for CODEFINDER tests.
"""
import pytest
from unittest.mock import Mock, patch


@pytest.fixture
def mock_db():
    """Create a mock database session."""
    db = Mock()
    return db


@pytest.fixture
def auth_headers(client):
    """Get authentication headers for protected endpoints."""
    login_response = client.post("/api/auth/login", json={
        "username": "admin",
        "password": "admin123"
    })
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}
