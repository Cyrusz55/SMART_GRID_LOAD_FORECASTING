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
import logging
import os

import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

from apps.schemas import (
    ForecastRequest,
    ForecastResponse,
    HourlyForecast,
    HealthResponse,
    ReadinessResponse,
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

logger = logging.getLogger("apps.routes")

# How far beyond the last real observation we are willing to forecast.
# The model only knows patterns up to ~Aug 2018; beyond this it confidently
# extrapolates nonsense, so we refuse rather than mislead.
MAX_FORECAST_AHEAD = pd.Timedelta(days=7)


@router.get("/health", response_model=HealthResponse)
def health_check():
    """Liveness check: is the process up?

    Deliberately cheap - it does NOT touch the model. Use /ready to decide
    whether the app can actually serve predictions.
    """
    return {"status": "ok"}


@router.get("/ready", response_model=ReadinessResponse)
def readiness_check():
    """Readiness check: can this instance actually serve a forecast?

    Returns 503 until the model is in memory, so an orchestrator (Docker,
    Render, k8s) will not route traffic into a 500. Health of the *model*,
    not just the process.
    """
    from apps.model_loader import _resolve_path, model_loaded

    path = _resolve_path()
    if not model_loaded():
        return JSONResponse(
            status_code=503,
            content={
                "status": "not-ready",
                "model_loaded": False,
                "model_path": path,
                "detail": (
                    "Model not in memory."
                    if os.path.exists(path)
                    else f"Model file not found: {path}"
                ),
            },
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "model_path": path,
        "detail": "Model loaded and cached.",
    }


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
        logger.error("Forecast failed - model or data missing: %s", e)
        raise HTTPException(
            status_code=503,
            detail="Model or historical data is unavailable.",
        )
    except HTTPException:
        raise  # already a clean client error - don't mask it
    except Exception as e:
        # Log the real cause server-side (with the class name so it is
        # greppable), then hand the client an opaque message. Echoing str(e)
        # used to leak file paths and internal state to any caller.
        logger.exception("Unhandled error in /predict (%s)", type(e).__name__)
        raise HTTPException(
            status_code=500,
            detail=(
                "Internal error while building the forecast. "
                "The failure has been logged server-side."
            ),
        ) from e


def _recursive_forecast(model, feat_cols, region_name, region_code, start_ts, horizon):
    """
    Recursive hourly forecast (same technique as notebook Cell 21):
    predict hour h, feed its prediction back as 'past', predict h+1, ...

    The running load series is kept in a PRE-ALLOCATED numpy buffer rather than
    grown with np.append(). The old np.append() call copied the whole history
    on every one of the 168 steps (O(n^2)); indexing into a fixed buffer is
    linear and produces identical values.
    """

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

    # Refuse dates too far past the end of the training data (item 1.9):
    # nothing statistical is left to extrapolate from out there.
    horizon_limit = last_real_ts + MAX_FORECAST_AHEAD
    if start_ts is not None and start_ts > horizon_limit:
        raise HTTPException(
            status_code=400,
            detail=(
                f"start_datetime {start_ts} is beyond the reliable forecast "
                f"horizon (latest supported start: {horizon_limit}). "
                f"The model is trained only up to ~{last_real_ts.date()}."
            ),
        )

    n_hist = len(df)
    # One buffer holds history + all forecasted hours: y[n_hist - 1] is the
    # last real value, and each prediction is written to the next free slot.
    y = np.empty(n_hist + horizon, dtype="float64")
    y[:n_hist] = df["MW"].to_numpy(dtype="float64")

    out = []
    ts = last_real_ts

    for step in range(horizon):
        ts = ts + pd.Timedelta(hours=1)
        # A view that ends at the last known value - same semantics as the old
        # "y" list, so the y[-1]/y[-24]/.../y[-720] lookups are unchanged.
        known = y[: n_hist + step]
        row = _build_future_row(ts, known, feat_cols)
        feat = pd.DataFrame([row])[feat_cols]
        pred_mw = float(model.predict(feat)[0])
        out.append(HourlyForecast(hour_ts=ts, predicted_mw=round(pred_mw, 2)))

        # Write the forecast into the buffer instead of reallocating the array.
        y[n_hist + step] = pred_mw

    return out


def _build_future_row(ts, y, feat_cols):
    """Given the running load series y, build all 26 features for time ts."""
    h, d, m = ts.hour, ts.dayofweek, ts.month

    # Dispatch table: feature name -> value. A dict beats a 26-branch if/elif
    # chain here because the model's own feature list drives the lookup, so a
    # missing branch must be visible. With the old if/elif there was no else,
    # and a feature the model expects but this table does not cover would
    # simply never be set - surfacing much later as a pandas KeyError on the
    # DataFrame construction, far from the actual cause.
    values = {
        # --- calendar / cyclical ---
        "hour": h,
        "day_of_week": d,
        "month": m,
        "year": ts.year,
        "is_weekend": int(d >= 5),
        "hour_sin": np.sin(2 * np.pi * h / 24),
        "hour_cos": np.cos(2 * np.pi * h / 24),
        "day_of_week_sin": np.sin(2 * np.pi * d / 7),
        "day_of_week_cos": np.cos(2 * np.pi * d / 7),
        "month_sin": np.sin(2 * np.pi * m / 12),
        "month_cos": np.cos(2 * np.pi * m / 12),
        "is_month_start": int(ts.day == 1),
        "is_month_end": int(ts.is_month_end),
        # Constant during training - see book2_loaded.ipynb cell 43. Passing 0
        # at inference MATCHES training; see item 1.2 in IMPROVEMENTS.md.
        "is_holiday": 0,
        "fourier_sin_1": np.sin(2 * np.pi * (h - 1) / 24),
        "fourier_cos_1": np.cos(2 * np.pi * (h - 1) / 24),
        "fourier_sin_2": np.sin(2 * np.pi * (h - 1) / 12),
        "fourier_cos_2": np.cos(2 * np.pi * (h - 1) / 12),
        # --- lags: NaN when there is not enough history yet ---
        "load_lag_1h": y[-1] if len(y) >= 1 else np.nan,
        "load_lag_24h": y[-24] if len(y) >= 24 else np.nan,
        "load_lag_168h": y[-168] if len(y) >= 168 else np.nan,
        "load_lag_720h": y[-720] if len(y) >= 720 else np.nan,
        # --- rolling stats ---
        "load_roll_mean_24h": y[-24:].mean(),
        "load_roll_std_24h": y[-24:].std(),
        "load_roll_mean_168h": y[-168:].mean(),
        "load_roll_std_168h": y[-168:].std(),
    }

    # Build in the model's own column order, and fail loudly HERE if the model
    # expects a feature this table does not know about.
    try:
        return {name: values[name] for name in feat_cols}
    except KeyError as exc:
        raise KeyError(
            f"feature {exc.args[0]!r} is expected by the model but is not "
            f"built by _build_future_row(); add it to the dispatch table"
        ) from exc


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
