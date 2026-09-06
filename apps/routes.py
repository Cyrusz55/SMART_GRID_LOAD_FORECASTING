"""
routes.py
=========
FastAPI routes for the load-forecasting app.

Maps to the heart-disease project's routes.py:
  GET  /health      -> liveness check
  POST /predict     -> build a future hourly forecast
  GET  /model-info  -> describe the loaded model

The heavy lifting (loading the model, recursive forecasting) lives in
machine_learning/machine_learning.py; this file only ties the HTTP layer
to that logic.
"""
import os

import pandas as pd
from fastapi import APIRouter, HTTPException

from apps.schemas import (
    ForecastRequest,
    ForecastResponse,
    HourlyForecast,
    HealthResponse,
)
from apps.model_loader import get_model
from machine_learning.machine_learning import load_feature_cols

# ---------------------------------------------------------------------------
# Quick in-code catalog of the 12 regions for mapping a name to its code.
# The model was trained on numeric codes 0..11 where 0=AEP, 1=COMED, etc.
# ---------------------------------------------------------------------------
REGION_NAMES = [
    "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ",
    "EKPC", "FE", "NI", "PJME", "PJMW", "PJM_Load",
]

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Simple endpoint used by load balancers / monitors."""
    return {"status": "ok"}


@router.post("/predict", response_model=ForecastResponse)
def forecast(req: ForecastRequest):
    """
    Forecast the load for a region over the requested horizon.

    NOTE ON DATE RANGE: the trained model only knows patterns up to the end
    of its training data (~Aug 2018). Reliable forecasts extend only a short
    way beyond the last real data point. Far-future dates (e.g. 2024) are
    NOT statistically meaningful with this model.
    """
    try:
        # 1) Get the CACHED model + the exact feature columns it expects.
        #    get_model() loads the model on first call, then reuses it, so the
        #    ~1GB file is only read from disk once.
        model = get_model()
        feat_cols = load_feature_cols()

        # 2) Resolve the requested region name to its numeric code.
        try:
            region_code = REGION_NAMES.index(req.region)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown region '{req.region}'. Valid: {REGION_NAMES}",
            )

        # 3) Build a recursive forecast one hour at a time.
        #    Real historical context for this region comes from the raw data
        #    that the model was trained on. (For a full production service this
        #    would read from the DB populated by scripts/load.py - here we
        #    demonstrate against the merged CSV path.)
        # ------------------------------------------------------------------
        forecasts = _recursive_forecast(
            model,
            feat_cols,
            region_name=req.region,
            region_code=region_code,
            start_ts=req.start_datetime,
            horizon=req.horizon_hours,
        )

        # 4) Wrap the result in the response schema.
        return ForecastResponse(
            region=req.region,
            forecasts=forecasts,
        )

    except FileNotFoundError as e:
        print(f"[DEBUG] FileNotFoundError: {e}")
        raise HTTPException(status_code=503, detail="Model file not found")
    except HTTPException:
        raise  # already a clean client error - don't mask it
    except Exception as e:
        print(f"[DEBUG] Exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


def _recursive_forecast(model, feat_cols, region_name, region_code, start_ts, horizon):
    """
    Recursive hourly forecast (same technique as notebook Cell 21):
    predict hour h, feed its prediction back as 'past', predict h+1, ...
    """
    import numpy as np

    # Read only the needed columns for this one region to limit RAM.
    raw_path = _resolve_raw_path()
    df = pd.read_csv(
        raw_path,
        usecols=["Datetime", "region", "MW"] + feat_cols,
    )
    df = df[df["region"] == region_name].copy()
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Start forecasting from the last REAL hour of this region's history.
    last_real_ts = df["Datetime"].iloc[-1]
    if start_ts is not None and start_ts <= last_real_ts:
        raise HTTPException(
            status_code=400,
            detail=(
                f"start_datetime {start_ts} is inside historical data "
                f"(last real hour ~{last_real_ts}). Give a future start time "
                "to forecast forward."
            ),
        )

    work = df[["Datetime", "MW"]].copy()
    y = work["MW"].values
    out = []
    ts = last_real_ts

    for _ in range(horizon):
        ts = ts + pd.Timedelta(hours=1)
        row = _build_future_row(ts, y, feat_cols)
        feat = pd.DataFrame([row])[feat_cols]
        pred_mw = float(model.predict(feat)[0])
        out.append(HourlyForecast(hour_ts=ts, predicted_mw=round(pred_mw, 2)))

        # Grow the running load series with the new forecasted value.
        y = np.append(y, pred_mw)

    return out


def _build_future_row(ts, y, feat_cols):
    """Given the running load series y, build all 26 features for time ts."""
    import numpy as np

    h, d, m = ts.hour, ts.dayofweek, ts.month
    row = {}

    # Calendar / cyclical features.
    for name in feat_cols:
        if name == "hour":          row[name] = h
        elif name == "day_of_week": row[name] = d
        elif name == "month":       row[name] = m
        elif name == "year":        row[name] = ts.year
        elif name == "is_weekend":  row[name] = int(d >= 5)
        elif name == "hour_sin":            row[name] = np.sin(2 * np.pi * h / 24)
        elif name == "hour_cos":            row[name] = np.cos(2 * np.pi * h / 24)
        elif name == "day_of_week_sin":     row[name] = np.sin(2 * np.pi * d / 7)
        elif name == "day_of_week_cos":     row[name] = np.cos(2 * np.pi * d / 7)
        elif name == "month_sin":           row[name] = np.sin(2 * np.pi * m / 12)
        elif name == "month_cos":           row[name] = np.cos(2 * np.pi * m / 12)
        elif name == "is_month_start":      row[name] = int(ts.day == 1)
        elif name == "is_month_end":        row[name] = int(ts.is_month_end)
        elif name == "is_holiday":          row[name] = 0
        elif name == "fourier_sin_1":       row[name] = np.sin(2 * np.pi * (h - 1) / 24)
        elif name == "fourier_cos_1":       row[name] = np.cos(2 * np.pi * (h - 1) / 24)
        elif name == "fourier_sin_2":       row[name] = np.sin(2 * np.pi * (h - 1) / 12)
        elif name == "fourier_cos_2":       row[name] = np.cos(2 * np.pi * (h - 1) / 12)
        elif name == "load_lag_1h":         row[name] = y[-1] if len(y) >= 1 else np.nan
        elif name == "load_lag_24h":        row[name] = y[-24] if len(y) >= 24 else np.nan
        elif name == "load_lag_168h":       row[name] = y[-168] if len(y) >= 168 else np.nan
        elif name == "load_lag_720h":       row[name] = y[-720] if len(y) >= 720 else np.nan
        elif name == "load_roll_mean_24h":  row[name] = y[-24:].mean()
        elif name == "load_roll_std_24h":   row[name] = y[-24:].std()
        elif name == "load_roll_mean_168h": row[name] = y[-168:].mean()
        elif name == "load_roll_std_168h":  row[name] = y[-168:].std()

    return row


def _resolve_raw_path():
    """Path to the merged CSV (fall back if configured elsewhere)."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    for rel in (
        "data_raw/merged_long_df.csv",
        "clean_data/merged_long_df_clean.csv",
    ):
        p = root / rel
        if p.exists():
            return str(p)
    raise FileNotFoundError("No merged/clean CSV found to build forecast context.")


@router.get("/model-info")
def model_info():
    """Describe which model file is in use (uses the cached loader)."""
    from apps.model_loader import _resolve_path, model_loaded

    path = _resolve_path()
    if not os.path.exists(path):
        return {"status": "no model found"}

    # Reuse the cached object if present; otherwise peek at the file type
    # without forcing a full load on this lightweight endpoint.
    if model_loaded():
        model = get_model()
    else:
        import joblib
        model = joblib.load(path)

    return {
        "model_type": type(model).__name__,
        "model_path": path,
        "in_memory": model_loaded(),
    }
