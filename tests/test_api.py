import pytest
from src import config


@pytest.mark.skipif(not config.ONNX_PATH.exists(),
                    reason="需先執行 python -m src.export_onnx 產生 ONNX 模型")
def test_health():
    from fastapi.testclient import TestClient
    from src.app import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"
