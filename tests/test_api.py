"""Run with:  pytest -q"""
import pytest
from fastapi.testclient import TestClient

from src.api.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:          # the `with` runs startup (loads the models)
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_machines(client):
    assert len(client.get("/machines").json()) == 100


def test_predict_shape(client):
    r = client.post("/predict", json={"unit": 35})
    assert r.status_code == 200
    d = r.json()
    assert 0 <= d["risk"] <= 1
    assert d["level"] in (1, 2, 3, 4)
    assert len(d["drivers"]) == 5


def test_replay_earlier_cycle_is_safer(client):
    early = client.post("/predict", json={"unit": 35, "cycle": 50}).json()
    late = client.post("/predict", json={"unit": 35}).json()
    assert early["cycle"] == 50 and early["risk"] < late["risk"]


def test_unknown_machine(client):
    assert client.post("/predict", json={"unit": 9999}).status_code == 404


def test_bad_input(client):
    assert client.post("/predict", json={"unit": 0}).status_code == 422


def test_history_logs_predictions(client):
    client.post("/predict", json={"unit": 35})
    assert len(client.get("/history", params={"unit": 35}).json()) >= 1
