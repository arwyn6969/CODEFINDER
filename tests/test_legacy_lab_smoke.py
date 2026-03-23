"""Collected smoke tests for the Legacy Exploratory Lab."""

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


def test_legacy_lab_gematria_smoke():
    response = client.post(
        "/api/research/gematria",
        json={"text": "Pepe"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    assert "english_standard" in data["results"]


def test_legacy_lab_transliteration_smoke():
    response = client.post(
        "/api/research/transliterate",
        json={"text": "PEPE"},
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    data = response.json()
    assert any(candidate["hebrew"] == "פפי" for candidate in data)


def test_legacy_lab_els_and_visualizer_smoke():
    response = client.post(
        "/api/research/els",
        json={
            "text": "The quick brown fox jumps over the lazy dog",
            "terms": ["fox"],
            "min_skip": 1,
            "max_skip": 10,
        },
        headers=_auth_headers(),
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["matches"]) > 0

    match = data["matches"][0]
    assert "start_index" in match
    assert "end_index" in match
    assert match["location"] == [match["start_index"], match["end_index"]]

    vis_response = client.post(
        "/api/research/els/visualize",
        json={
            "text": "The quick brown fox jumps over the lazy dog",
            "center_index": match["start_index"],
            "skip": match["skip"],
            "rows": 5,
            "cols": 5,
            "term_length": len(match["term"]),
        },
        headers=_auth_headers(),
    )

    assert vis_response.status_code == 200
    vis_data = vis_response.json()
    assert "grid" in vis_data
    assert "highlights" in vis_data
    assert len(vis_data["highlights"]) == len(match["term"])
