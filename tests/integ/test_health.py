import pytest
from fastapi.testclient import TestClient

pytestmark = pytest.mark.skip(reason="requires heavy model imports")

from main import app

client = TestClient(app)

def test_healthz():
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
