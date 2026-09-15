"""
tests/test_api.py
=================
API-surface tests with a STUBBED model (IMPROVEMENTS.md item 2.3).

The real Random Forest is 1.05 GiB and `_recursive_forecast` reads a 342 MiB
CSV. Neither belongs in a unit test. So here we:
  * monkeypatch `get_model`  -> FakeRegressor
  * monkeypatch the CSV read -> tiny in-memory frame

That keeps the HTTP layer, the validation, the error mapping and the response
schema under test, while staying fast and offline.
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from apps import routes
from apps.main import app


@pytest.fixture
def client(monkeypatch, fake_model, feature_cols):
    """TestClient with the model AND the historical CSV both stubbed out.

    Returns (client, fake_model) so tests can inspect how the model was called.
    """
    monkeypatch.setattr(routes, "get_model", lambda: fake_model)

    # A short but valid history: 800 hourly points ending 2018-08-01 00:00.
    end = pd.Timestamp("2018-08-01 00:00:00")
    stamps = pd.date_range(end=end, periods=800, freq="h")
    hist = pd.DataFrame({
        "Datetime": stamps,
        "region": "AEP",
        "MW": np.linspace(10000.0, 12000.0, 800),
    })
    # Add the feature columns so `usecols=...` in _recursive_forecast resolves.
    for col in feature_cols:
        hist[col] = 0.0

    monkeypatch.setattr(
        routes.pd, "read_csv", lambda *a, **k: hist.copy()
    )
    monkeypatch.setattr(routes, "_resolve_raw_path", lambda: "/fake/hist.csv")

    # IMPORTANT: do NOT use `with TestClient(app)`. Entering the context runs
    # the app's lifespan handler, which warms the model -> unpickles the
    # 1.05 GiB RandomForest and hangs the suite. A bare TestClient still sends
    # real requests but skips startup, and nothing here needs startup state
    # because get_model is monkeypatched above.
    yield TestClient(app), fake_model


# ---------------------------------------------------------------------------
# Liveness / readiness
# ---------------------------------------------------------------------------
def test_health_is_always_ok(client):
    c, _ = client
    r = c.get("/api/v1/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_ready_reports_ready_when_model_in_memory(client):
    c, _ = client
    r = c.get("/api/v1/ready")
    body = r.json()
    # The model fixture may or may not be primed; both are valid answers,
    # but the body must always carry the four contract fields.
    assert r.status_code in (200, 503)
    assert set(body) == {"status", "model_loaded", "model_path", "detail"}
    assert body["status"] in ("ready", "not-ready")
    assert isinstance(body["model_loaded"], bool)


def test_root_returns_json_message(client):
    """Item 1.7: there is no frontend/index.html, so this is JSON only."""
    c, _ = client
    r = c.get("/")
    assert r.status_code == 200
    assert "message" in r.json()


# ---------------------------------------------------------------------------
# /predict happy path
# ---------------------------------------------------------------------------
def test_predict_returns_forecast_response_shape(client):
    c, fake = client
    start = "2018-08-01T01:00:00"
    r = c.post("/api/v1/predict", json={
        "region": "AEP",
        "start_datetime": start,
        "horizon_hours": 5,
    })
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["region"] == "AEP"
    assert len(body["forecasts"]) == 5
    for item in body["forecasts"]:
        assert set(item) == {"datetime", "predicted_mw"}
        assert item["predicted_mw"] == pytest.approx(fake.value)


def test_predict_hours_are_consecutive_and_start_one_hour_ahead(client):
    """Forecast begins at last_real_ts + 1h and steps by exactly one hour."""
    c, _ = client
    r = c.post("/api/v1/predict", json={
        "region": "AEP",
        "start_datetime": "2018-08-01T01:00:00",
        "horizon_hours": 3,
    })
    hours = [datetime.fromisoformat(f["datetime"]) for f in r.json()["forecasts"]]
    assert hours[0] == datetime(2018, 8, 1, 1, 0)
    assert hours == [hours[0] + timedelta(hours=i) for i in range(3)]


def test_predict_materialises_all_26_features(client):
    """The model must receive exactly the 26 trained columns, in order."""
    c, fake = client
    c.post("/api/v1/predict", json={
        "region": "AEP",
        "start_datetime": "2018-08-01T01:00:00",
        "horizon_hours": 2,
    })
    assert len(fake.calls) == 2  # one predict() per forecast hour
    assert len(fake.calls[0]) == 26
    assert fake.calls[0] == fake.calls[1]  # same order every step


# ---------------------------------------------------------------------------
# /predict error mapping
# ---------------------------------------------------------------------------
def test_predict_rejects_start_inside_history(client):
    """Item: start_datetime <= last real hour -> 400 with a helpful message."""
    c, _ = client
    r = c.post("/api/v1/predict", json={
        "region": "AEP",
        "start_datetime": "2018-07-15T00:00:00",
        "horizon_hours": 3,
    })
    assert r.status_code == 400
    assert "inside historical data" in r.json()["detail"]


def test_predict_rejects_far_future_start(client):
    """Item 1.9: beyond last_real_ts + 7 days -> 400."""
    c, _ = client
    r = c.post("/api/v1/predict", json={
        "region": "AEP",
        "start_datetime": "2024-01-01T00:00:00",
        "horizon_hours": 3,
    })
    assert r.status_code == 400
    assert "reliable forecast horizon" in r.json()["detail"]


def test_predict_accepts_start_at_the_horizon_edge(client):
    """Exactly last_real_ts + 7 days is still allowed (boundary is inclusive)."""
    c, _ = client
    r = c.post("/api/v1/predict", json={
        "region": "AEP",
        "start_datetime": "2018-08-08T00:00:00",
        "horizon_hours": 2,
    })
    assert r.status_code == 200, r.text


def test_predict_rejects_unknown_region_at_the_schema_layer(client):
    c, _ = client
    r = c.post("/api/v1/predict", json={
        "region": "NOPE",
        "start_datetime": "2018-08-01T01:00:00",
    })
    assert r.status_code == 422  # pydantic literal validation


def test_predict_rejects_bad_horizon(client):
    c, _ = client
    for horizon in (0, 169):
        r = c.post("/api/v1/predict", json={
            "region": "AEP",
            "start_datetime": "2018-08-01T01:00:00",
            "horizon_hours": horizon,
        })
        assert r.status_code == 422


def test_internal_errors_do_not_leak_internals(client, monkeypatch):
    """Item 1.10: a 500 must be opaque, not echo str(exception)."""
    c, _ = client

    def boom(*a, **k):
        raise ValueError("SECRET PATH /etc/passwd and internal state")

    monkeypatch.setattr(routes, "_build_future_row", boom)
    r = c.post("/api/v1/predict", json={
        "region": "AEP",
        "start_datetime": "2018-08-01T01:00:00",
        "horizon_hours": 1,
    })
    assert r.status_code == 500
    detail = r.json()["detail"]
    assert "SECRET PATH" not in detail
    assert "/etc/passwd" not in detail
    assert "Internal error" in detail
