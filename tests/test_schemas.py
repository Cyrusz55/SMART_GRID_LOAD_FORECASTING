"""
tests/test_schemas.py
=====================
Validation-only tests for apps/schemas.py (IMPROVEMENTS.md item 2.2).

These need NO model and NO data - they run in milliseconds and are the
fastest possible signal that the API contract has not drifted.
"""
from datetime import datetime

import pytest
from pydantic import ValidationError

from apps.schemas import (
    REGION_NAMES,
    ForecastRequest,
    ForecastResponse,
    HealthResponse,
    HourlyForecast,
    ReadinessResponse,
)


# ---------------------------------------------------------------------------
# Region handling
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("region", [
    "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ",
    "EKPC", "FE", "NI", "PJME", "PJMW", "PJM_Load",
])
def test_every_region_is_accepted(region):
    """All 12 PJM regions are valid."""
    req = ForecastRequest(region=region, start_datetime="2018-06-15T14:00:00")
    assert req.region == region


def test_region_list_has_12_unique_entries():
    assert len(REGION_NAMES) == 12
    assert len(set(REGION_NAMES)) == 12


@pytest.mark.parametrize("bad_region", ["aep", "AEPP", "PJM", "", "PJM_Load "])
def test_invalid_region_is_rejected(bad_region):
    """Case matters and whitespace is not stripped - reject, don't guess."""
    with pytest.raises(ValidationError):
        ForecastRequest(region=bad_region, start_datetime="2018-06-15T14:00:00")


# ---------------------------------------------------------------------------
# horizon_hours bounds (ge=1, le=168)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("horizon", [0, -1, 169, 1000])
def test_horizon_out_of_range_is_rejected(horizon):
    with pytest.raises(ValidationError):
        ForecastRequest(
            region="AEP",
            start_datetime="2018-06-15T14:00:00",
            horizon_hours=horizon,
        )


@pytest.mark.parametrize("horizon", [1, 24, 168])
def test_horizon_in_range_is_accepted(horizon):
    req = ForecastRequest(
        region="AEP",
        start_datetime="2018-06-15T14:00:00",
        horizon_hours=horizon,
    )
    assert req.horizon_hours == horizon


def test_horizon_defaults_to_24():
    req = ForecastRequest(region="AEP", start_datetime="2018-06-15T14:00:00")
    assert req.horizon_hours == 24


# ---------------------------------------------------------------------------
# start_datetime parsing
# ---------------------------------------------------------------------------
def test_start_datetime_parses_iso_string():
    req = ForecastRequest(region="AEP", start_datetime="2018-06-15T14:00:00")
    assert req.start_datetime == datetime(2018, 6, 15, 14, 0, 0)


def test_start_datetime_is_required():
    with pytest.raises(ValidationError):
        ForecastRequest(region="AEP")


def test_start_datetime_rejects_garbage():
    with pytest.raises(ValidationError):
        ForecastRequest(region="AEP", start_datetime="not-a-date")


# ---------------------------------------------------------------------------
# HourlyForecast alias round-trip ('datetime' on the wire, 'hour_ts' in Python)
# ---------------------------------------------------------------------------
def test_hourly_forecast_accepts_wire_alias():
    hf = HourlyForecast(**{"datetime": "2018-06-15T15:00:00", "predicted_mw": 1234.5})
    assert hf.hour_ts == datetime(2018, 6, 15, 15, 0, 0)
    assert hf.predicted_mw == 1234.5


def test_hourly_forecast_accepts_python_name():
    """populate_by_name=True means the internal name works too."""
    hf = HourlyForecast(hour_ts="2018-06-15T15:00:00", predicted_mw=1234.5)
    assert hf.hour_ts == datetime(2018, 6, 15, 15, 0, 0)


def test_hourly_forecast_serialises_back_to_alias():
    """The wire format must be 'datetime', not 'hour_ts'."""
    hf = HourlyForecast(hour_ts="2018-06-15T15:00:00", predicted_mw=1234.5)
    dumped = hf.model_dump(by_alias=True)
    assert "datetime" in dumped
    assert "hour_ts" not in dumped


def test_forecast_response_round_trips():
    resp = ForecastResponse(
        region="AEP",
        forecasts=[
            {"datetime": "2018-06-15T15:00:00", "predicted_mw": 100.0},
            {"datetime": "2018-06-15T16:00:00", "predicted_mw": 200.0},
        ],
    )
    assert len(resp.forecasts) == 2
    payload = resp.model_dump(by_alias=True)
    assert payload["region"] == "AEP"
    assert payload["forecasts"][0]["datetime"] == datetime(2018, 6, 15, 15, 0)
    assert payload["forecasts"][1]["predicted_mw"] == 200.0


# ---------------------------------------------------------------------------
# Health / readiness contracts
# ---------------------------------------------------------------------------
def test_health_only_accepts_ok():
    assert HealthResponse(status="ok").status == "ok"
    with pytest.raises(ValidationError):
        HealthResponse(status="degraded")


@pytest.mark.parametrize("state", ["ready", "not-ready"])
def test_readiness_accepts_both_states(state):
    r = ReadinessResponse(
        status=state,
        model_loaded=(state == "ready"),
        model_path="/tmp/model.joblib",
        detail="whatever",
    )
    assert r.status == state


def test_readiness_rejects_unknown_state():
    with pytest.raises(ValidationError):
        ReadinessResponse(
            status="maybe",
            model_loaded=False,
            model_path="/tmp/model.joblib",
            detail="x",
        )
