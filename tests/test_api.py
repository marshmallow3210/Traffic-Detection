from fastapi.testclient import TestClient
from src.stream import app


def test_feed_endpoint_exists():
    with TestClient(app) as client:
        r = client.get("/feed")
        assert r.status_code == 200


def test_stats_endpoint():
    with TestClient(app) as client:
        r = client.get("/stats")
        assert r.status_code == 200
        assert "stats" in r.json()