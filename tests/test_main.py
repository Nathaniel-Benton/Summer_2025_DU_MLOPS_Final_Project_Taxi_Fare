"""
Integration tests for the FastAPI endpoints in main.py.

main.py attempts to load a real model from W&B and connect to a real
DynamoDB table at import time. Both are wrapped in try/except and fall
back to None on failure, which is exactly what happens in a CI
environment with no credentials configured — so importing `main` here
is safe and won't hit any real network calls.

Each test then monkeypatches `main.model` and/or `main.prediction_table`
with fakes to exercise both the "healthy" and "degraded" code paths
without depending on any external service.
"""
from unittest.mock import MagicMock

from fastapi.testclient import TestClient

import main

client = TestClient(main.app)


class FakeModel:
    """Stands in for the real XGBoost pipeline."""

    def predict(self, input_df):
        # Always return a single, predictable fare
        return [17.50]


# --- /health ---

def test_health_returns_200_when_model_loaded(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_health_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main, "model", None)

    response = client.get("/health")

    assert response.status_code == 503


# --- /predict ---

VALID_TRIP_PAYLOAD = {
    "trip_distance": 3.5,
    "passenger_count": 1,
    "PULocationID": 142,
    "DOLocationID": 236,
    "RatecodeID": 1,
}


def test_predict_returns_503_when_model_not_loaded(monkeypatch):
    monkeypatch.setattr(main, "model", None)

    response = client.post("/predict", json=VALID_TRIP_PAYLOAD)

    assert response.status_code == 503


def test_predict_returns_prediction_and_id_when_model_loaded(monkeypatch):
    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "prediction_table", None)  # DB down, shouldn't matter

    response = client.post("/predict", json=VALID_TRIP_PAYLOAD)

    assert response.status_code == 200
    body = response.json()
    assert body["predicted_fare_amount"] == 17.50
    assert "prediction_id" in body
    assert len(body["prediction_id"]) > 0


def test_predict_logs_to_dynamodb_when_table_available(monkeypatch):
    fake_table = MagicMock()
    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "prediction_table", fake_table)

    response = client.post("/predict", json=VALID_TRIP_PAYLOAD)

    assert response.status_code == 200
    fake_table.put_item.assert_called_once()

    # Sanity-check the logged item contains the fields the dashboard relies on
    logged_item = fake_table.put_item.call_args.kwargs["Item"]
    assert logged_item["predicted_fare"] == 17.50
    assert logged_item["trip_distance"] == 3.5
    assert "latency_ms" in logged_item
    assert "timestamp" in logged_item


def test_predict_still_succeeds_if_dynamodb_write_fails(monkeypatch):
    fake_table = MagicMock()
    fake_table.put_item.side_effect = Exception("simulated DynamoDB outage")

    monkeypatch.setattr(main, "model", FakeModel())
    monkeypatch.setattr(main, "prediction_table", fake_table)

    response = client.post("/predict", json=VALID_TRIP_PAYLOAD)

    # A logging failure should never break the prediction response itself
    assert response.status_code == 200
    assert "predicted_fare_amount" in response.json()


def test_predict_rejects_malformed_payload():
    response = client.post("/predict", json={"trip_distance": "not_a_number"})

    assert response.status_code == 422  # FastAPI/pydantic validation error


# --- /feedback ---

def test_feedback_returns_503_when_dynamodb_unavailable(monkeypatch):
    monkeypatch.setattr(main, "prediction_table", None)

    response = client.post("/feedback", json={
        "prediction_id": "some-id",
        "actual_fare": 20.0,
    })

    assert response.status_code == 503


def test_feedback_succeeds_when_dynamodb_available(monkeypatch):
    fake_table = MagicMock()
    monkeypatch.setattr(main, "prediction_table", fake_table)

    response = client.post("/feedback", json={
        "prediction_id": "some-id",
        "actual_fare": 20.0,
    })

    assert response.status_code == 200
    assert response.json()["status"] == "feedback recorded"
    fake_table.update_item.assert_called_once()


def test_feedback_returns_500_when_dynamodb_update_fails(monkeypatch):
    fake_table = MagicMock()
    fake_table.update_item.side_effect = Exception("simulated failure")
    monkeypatch.setattr(main, "prediction_table", fake_table)

    response = client.post("/feedback", json={
        "prediction_id": "some-id",
        "actual_fare": 20.0,
    })

    assert response.status_code == 500