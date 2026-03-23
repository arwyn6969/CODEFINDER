from pathlib import Path

from fastapi.testclient import TestClient

from app.api.main import create_app


def test_app_boots_without_frontend_build(tmp_path):
    app = create_app(frontend_build_dir=tmp_path / "missing-build")

    with TestClient(app) as client:
        health = client.get("/api/health")
        root = client.get("/")

    assert health.status_code == 200
    assert root.status_code == 200
    assert "Frontend not built" in root.text


def test_app_boots_with_partial_frontend_build(tmp_path):
    build_dir = tmp_path / "partial-build"
    build_dir.mkdir()
    (build_dir / "index.html").write_text("<html><body>partial frontend</body></html>")

    app = create_app(frontend_build_dir=build_dir)

    with TestClient(app) as client:
        health = client.get("/api/health")
        root = client.get("/")

    assert health.status_code == 200
    assert root.status_code == 200
    assert "partial frontend" in root.text
