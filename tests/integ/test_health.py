import pytest
pytest.skip(reason="requires heavy model imports", allow_module_level=True)

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
