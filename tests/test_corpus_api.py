"""
API tests for the corpus registry surface.
"""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.api.dependencies import get_database
from app.api.main import app
from app.core.database import Base


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite://",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(db_session):
    app.dependency_overrides[get_database] = lambda: db_session
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        app.dependency_overrides.clear()


def auth_headers(client: TestClient) -> dict[str, str]:
    login_response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_corpus_ingest_and_summary_routes(client):
    headers = auth_headers(client)

    ingest_response = client.post(
        "/api/corpus/ingest",
        json={
            "records": [
                {
                    "title": "Vita, Non Vita",
                    "work_key": "work:vita-non-vita",
                    "witness_key": "witness:gdz",
                    "source_key": "source:gdz",
                    "language": "German",
                    "script": "fraktur_blackletter",
                    "printer": "Christoff Krausen",
                    "place": "Kempten",
                    "date_range": "1609",
                    "print_or_manuscript": "printed",
                    "image_set_key": "images:gdz"
                }
            ]
        },
        headers=headers,
    )
    summary_response = client.get("/api/corpus/summary", headers=headers)

    assert ingest_response.status_code == 200
    assert summary_response.status_code == 200
    assert summary_response.json()["witness_count"] == 1


def test_corpus_bootstrap_and_candidate_routes(client):
    headers = auth_headers(client)

    bootstrap_response = client.post(
        "/api/corpus/bootstrap/current-sources",
        json={"config_path": str(ROOT / "data/sources/config.yaml")},
        headers=headers,
    )
    build_response = client.post(
        "/api/corpus/candidates/build",
        json={"lane": "printed_european"},
        headers=headers,
    )
    list_response = client.get("/api/corpus/candidates", headers=headers)

    assert bootstrap_response.status_code == 200
    assert build_response.status_code == 200
    assert list_response.status_code == 200
    assert bootstrap_response.json()["records_prepared"] >= 8
    assert "candidates" in list_response.json()


def test_corpus_review_route_returns_404_for_missing_edge(client):
    headers = auth_headers(client)

    response = client.post(
        "/api/corpus/reviews",
        json={
            "edge_id": 999,
            "vocabulary": "inconclusive",
            "confidence": 0.5,
            "evidence_basis": "manual review",
            "safe_for_external": False,
        },
        headers=headers,
    )

    assert response.status_code == 404
