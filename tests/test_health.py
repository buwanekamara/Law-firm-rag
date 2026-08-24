"""Smoke test: the app boots and can see the contracts folder."""

from app.api import app
from fastapi.testclient import TestClient

client = TestClient(app)


def test_health_ok():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_all_five_contracts_are_visible():
    """Guards against the .PDF/.pdf case trap in the contracts folder."""
    body = client.get("/health").json()
    assert body["contracts_found"] == 5, body["contracts"]
