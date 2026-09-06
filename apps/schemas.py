"""
schemas.py
==========
Pydantic request/response models for the load-forecasting API.

These validate what comes IN (a forecast request) and define what goes OUT
(a list of hourly MW predictions), so the app never receives junk input.

Note: unlike the heart-disease app, a user does not type 26 ML features.
Load forecasting is driven by TIME - the caller chooses a region and a
start time, and the model predicts the load for the next N hours.
"""
from __future__ import annotations

from datetime import datetime as dt
from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Region codes used throughout the dataset (see merged_long_df / book notebooks).
# Index of each name = its numeric code (0=AEP, 1=COMED, ...).
# ---------------------------------------------------------------------------
REGION_NAMES = [
    "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ",
    "EKPC", "FE", "NI", "PJME", "PJMW", "PJM_Load",
]


class ForecastRequest(BaseModel):
    """What the caller sends to ask for a forecast."""

    region: Literal[
        "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ",
        "EKPC", "FE", "NI", "PJME", "PJMW", "PJM_Load",
    ] = Field(
        ...,
        description="PJM region to forecast (e.g. 'AEP', 'COMED', 'PJM_Load').",
    )

    start_datetime: dt = Field(
        ...,
        description="First hour to forecast, e.g. 2018-06-15 14:00:00. "
                    "Should be within the historical data range (c. 1998-2018).",
    )

    horizon_hours: int = Field(
        default=24,
        ge=1,
        le=168,
        description="How many future hours to forecast (1-168, default 24).",
    )


class HourlyForecast(BaseModel):
    """One forecasted hour."""
    # Field is named 'datetime' on the wire for the caller, but the Python
    # attribute is 'hour_ts' so it does not clash with the imported type.
    hour_ts: dt = Field(
        ...,
        alias="datetime",
        description="The hour being forecast.",
    )
    predicted_mw: float = Field(
        ...,
        description="Predicted electrical load in megawatts for that hour.",
    )

    model_config = {"populate_by_name": True}


class ForecastResponse(BaseModel):
    """What the API returns: a list of hourly MW forecasts."""

    region: str = Field(..., description="Region that was forecast.")
    forecasts: list[HourlyForecast] = Field(
        ...,
        description="Hourly load forecast for the requested horizon.",
    )


class HealthResponse(BaseModel):
    """Simple liveness/health check for the app endpoint."""
    status: Literal["ok"] = Field(..., description="Service status.")
