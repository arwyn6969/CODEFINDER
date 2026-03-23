"""Route compatibility tests for canonical and legacy API prefixes."""

from fastapi.testclient import TestClient

from app.api.main import app


_client = None


def get_client():
    global _client
    if _client is None:
        _client = TestClient(app)
    return _client


class LazyClient:
    def __getattr__(self, name):
        return getattr(get_client(), name)


client = LazyClient()


def _auth_headers():
    login_response = client.post(
        "/api/auth/login",
        json={"username": "admin", "password": "admin123"},
    )
    token = login_response.json()["access_token"]
    return {"Authorization": f"Bearer {token}"}


def test_health_legacy_alias_maps_to_canonical_health():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"


def test_docs_legacy_alias_redirects_to_api_docs():
    response = client.get("/docs", follow_redirects=False)

    assert response.status_code in (307, 308)
    assert response.headers.get("location") == "/api/docs"


def test_auth_login_available_on_legacy_v1_prefix():
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "admin123"},
    )

    assert response.status_code == 200
    assert "access_token" in response.json()


def test_research_gematria_available_on_canonical_and_legacy_prefixes():
    for path in ("/api/research/gematria", "/api/v1/research/gematria"):
        response = client.post(
            path,
            json={"text": "Francis Bacon"},
            headers=_auth_headers(),
        )

        assert response.status_code == 200


def test_research_transliteration_available_on_canonical_and_legacy_prefixes():
    for path in ("/api/research/transliterate", "/api/v1/research/transliterate"):
        response = client.post(
            path,
            json={"text": "PEPE"},
            headers=_auth_headers(),
        )

        assert response.status_code == 200


def test_research_els_available_on_canonical_and_legacy_prefixes():
    payload = {
        "text": "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG",
        "terms": ["FOX"],
        "min_skip": 1,
        "max_skip": 5,
    }

    for path in ("/api/research/els", "/api/v1/research/els"):
        response = client.post(
            path,
            json=payload,
            headers=_auth_headers(),
        )

        assert response.status_code == 200


def test_research_els_visualization_available_on_canonical_and_legacy_prefixes():
    payload = {
        "text": "THEQUICKBROWNFOXJUMPSOVERTHELAZYDOG",
        "center_index": 13,
        "skip": 1,
        "rows": 5,
        "cols": 5,
        "term_length": 3,
    }

    for path in ("/api/research/els/visualize", "/api/v1/research/els/visualize"):
        response = client.post(
            path,
            json=payload,
            headers=_auth_headers(),
        )

        assert response.status_code == 200


def test_research_geometry_alias_is_mounted_on_canonical_and_legacy_prefixes():
    for path in ("/api/research/geometry/999999", "/api/v1/research/geometry/999999"):
        response = client.get(
            path,
            headers=_auth_headers(),
        )

        assert response.status_code == 404


def test_analysis_geometry_route_remains_available():
    response = client.get(
        "/api/analysis/999999/geometric",
        headers=_auth_headers(),
    )

    assert response.status_code == 404


def test_relationships_network_route_mounted_on_both_prefixes():
    payload = {"document_ids": [1, 2]}

    canonical = client.post("/api/relationships/network", json=payload)
    legacy = client.post("/api/v1/relationships/network", json=payload)

    assert canonical.status_code != 404
    assert legacy.status_code != 404
