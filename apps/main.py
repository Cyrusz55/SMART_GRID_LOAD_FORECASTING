"""
main.py
=======
FastAPI application entrypoint for the Smart Grid Load Forecasting API.

Mirrors the heart-disease project's main.py:
  - Create the FastAPI app
  - Enable CORS
  - Mount the routes under /api/v1
  - Ensure the model is available on startup
  - Serve a simple root landing page (if a frontend exists)

Differences from the heart project:
  - The model is ALREADY trained and saved locally (models/best_rf_clean.joblib),
    so there is no HuggingFace download and no auto-training on startup.
  - A scheduler is not wired in here; forecasting is request-driven.
"""
import os

import pandas as pd  # noqa: F401  (left available, mirroring the heart main.py)
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from apps.routes import router

# ---------------------------------------------------------------------------
# Optional: if a periodic (cron) scheduler exists, import + start it.
# The heart project imported start_scheduler from a 'scheduler' module.
# Your project has none yet, so this is intentionally left out - uncomment
# only once you build a scheduler (e.g. nightly retraining).
# ---------------------------------------------------------------------------
# from scheduler import start_scheduler

load_dotenv()

app = FastAPI(
    title="Smart Grid Load Forecasting API",
    description=(
        "Forecast hourly electrical load (MW) for PJM regions using an "
        "already-trained Random Forest model."
    ),
    version="1.0.0",
)

# ---------------------------------------------------------------------------
# CORS - allow specific origins only (never '*' without a reason).
# ---------------------------------------------------------------------------
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    # Reflex frontend dev server (frontend/load_forecast.py runs on port 3000).
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount the endpoint router under the /api/v1 prefix.
app.include_router(router, prefix="/api/v1")


@app.on_event("startup")
def startup_event():
    """
    Ensure the model file exists when the API starts.

    Unlike the heart project (which downloads from HuggingFace or trains on
    first boot), our model is trained and stored locally - so we just verify
    it is present. If it is missing, we load it now (so the first /predict
    request does not trigger a slow, unexpected load).
    """
    from apps.model_loader import load_model

    os.makedirs("models", exist_ok=True)

    try:
        load_model()  # loads models/best_rf_clean.joblib into the cache
        print("[startup] Model loaded and cached.")
    except FileNotFoundError as e:
        print(f"[startup] WARNING: model not available: {e}")
        print("[startup] Run the training notebook (cyrus1) to produce "
              "models/best_rf_clean.joblib, then restart the API.")
    except Exception as e:
        print(f"[startup] Could not load model: {e}")


@app.get("/")
def root():
    """Root landing page (optional frontend) or a simple JSON message."""
    html_path = "frontend/index.html"

    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"message": "Smart Grid Load Forecasting API is running"}
