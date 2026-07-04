from fastapi.testclient import TestClient
from src.stream import app

def test_feed_returns_503_before_stream_starts():
    with TestClient(app) as client:
        r = client.get("/feed")
        assert r.status_code == 503
        
def test_stats_endpoint():
    with TestClient(app) as client:
        r = client.get("/stats")
        assert r.status_code == 200
        assert "stats" in r.json()

def test_detect_endpoint():
    with TestClient(app) as client:
        r = client.get("/detect")
        assert r.status_code == 200
        body = r.json()
        assert "detections" in body and "count" in body