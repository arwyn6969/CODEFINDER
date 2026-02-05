import asyncio
import sys
import os
from sqlalchemy.orm import Session
from fastapi.testclient import TestClient

# Add app to path
sys.path.append(os.getcwd())

from app.main import app
from app.core.database import get_db
from app.models.database_models import Document, Page

client = TestClient(app)

def test_gematria():
    print("Testing Gematria...")
    response = client.post("/api/research/gematria", json={"text": "Pepe"})
    assert response.status_code == 200
    data = response.json()
    assert "english_standard" in data
    print("✅ Gematria OK")

def test_els_flow():
    print("Testing ELS Flow...")
    # 1. Search
    response = client.post("/api/research/els", json={
        "text": "The quick brown fox jumps over the lazy dog",
        "terms": ["fox"],
        "min_skip": 1,
        "max_skip": 10
    })
    assert response.status_code == 200
    data = response.json()
    assert len(data['matches']) > 0
    match = data['matches'][0]
    print(f"✅ ELS Search OK (Found '{match['term']}')")
    
    # 2. Visualize
    print("Testing ELS Visualization...")
    vis_response = client.post("/api/research/els/visualize", json={
        "text": "The quick brown fox jumps over the lazy dog",
        "center_index": match['start_index'],
        "skip": match['skip'],
        "rows": 5,
        "cols": 5,
        "term_length": len(match['term'])
    })
    assert vis_response.status_code == 200
    vis_data = vis_response.json()
    assert "grid" in vis_data
    assert "highlights" in vis_data
    assert len(vis_data['highlights']) == len(match['term'])
    print("✅ ELS Visualization OK")

def test_documents_flow():
    print("Testing Documents Flow...")
    # 1. List (should be empty initially or have existing)
    response = client.get("/api/documents/")
    assert response.status_code == 200
    print("✅ Document List OK")
    
    # 2. Upload Dummy (Mocking file upload effectively requires careful setup, omitting for speed/safety in this script 
    # unless we want to use a real tmp file. Let's assume list/content get works if docs exist)
    
def test_network_analysis():
    print("Testing Network Analysis (Mock)...")
    # Need at least 2 docs. If not enough, it returns 400, which is a valid API response.
    # We'll just check if the endpoint is reachable.
    response = client.post("/api/relationships/network", json={"document_ids": [1, 2]})
    # It might fail with 400 (not enough docs) or 404 (docs not found) or 200.
    # We accept 200 or 400/404 as "endpoint is alive". 500 would be bad.
    assert response.status_code in [200, 400, 404]
    print(f"✅ Network Endpoint Alive (Status: {response.status_code})")

if __name__ == "__main__":
    try:
        test_gematria()
        test_els_flow()
        test_documents_flow()
        test_network_analysis()
        print("\n🎉 ALL SYSTEM CHECKS PASSED")
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        sys.exit(1)
