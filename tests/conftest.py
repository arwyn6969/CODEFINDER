"""
Pytest configuration and shared fixtures for CODEFINDER tests.
"""
import pytest
from unittest.mock import Mock
import inspect


def _maybe_patch_httpx_client_for_starlette():
    """
    Compatibility shim for Starlette TestClient with httpx>=0.28.
    Starlette 0.36 passes an `app=` argument that newer httpx.Client
    constructors no longer accept.
    """
    try:
        import httpx
    except ModuleNotFoundError:
        return None, None

    params = inspect.signature(httpx.Client.__init__).parameters
    if "app" in params:
        return None, None

    original_init = httpx.Client.__init__

    def patched_init(self, *args, app=None, **kwargs):
        return original_init(self, *args, **kwargs)

    httpx.Client.__init__ = patched_init
    return httpx, original_init


@pytest.fixture(scope="session", autouse=True)
def httpx_starlette_compat_patch():
    """
    Apply/restore the Starlette/httpx compatibility shim for the test session.
    """
    httpx_mod, original_init = _maybe_patch_httpx_client_for_starlette()
    try:
        yield
    finally:
        if httpx_mod is not None and original_init is not None:
            httpx_mod.Client.__init__ = original_init


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
