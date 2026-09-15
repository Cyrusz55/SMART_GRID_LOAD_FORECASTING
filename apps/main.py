"""
main.py
=======
FastAPI application entrypoint for the Smart Grid Load Forecasting API.

Mirrors the heart-disease project's main.py:
  - Create the FastAPI app
  - Enable CORS
  - Mount the routes under /api/v1
  - Ensure the model is available on startup (via the lifespan handler)
  - Serve a simple root landing page

Differences from the heart project:
  - The model is ALREADY trained and saved locally (models/best_rf_clean.joblib),
    so there is no HuggingFace download and no auto-training on startup.
  - A scheduler is not wired in here; forecasting is request-driven.
  - The frontend is a separate Reflex app under frontend/ (no static HTML here).
"""
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from apps.routes import router

# ---------------------------------------------------------------------------
# Optional: if a periodic (cron) scheduler exists, import + start it.
# The heart project imported start_scheduler from a 'scheduler' module.
# Your project has none yet, so this is intentionally left out - uncomment
# only once you build a scheduler (e.g. nightly retraining).
# ---------------------------------------------------------------------------
# from scheduler import start_scheduler

load_dotenv()

logger = logging.getLogger("apps.main")

# If true (the default), a model that fails to load is a hard startup error.
# Set EAGER_MODEL_LOAD=false to boot the API anyway - /api/v1/ready then
# reports 503 and the model is loaded on first /predict instead.
EAGER_MODEL_LOAD = os.getenv("EAGER_MODEL_LOAD", "true").strip().lower() not in (
    "0", "false", "no",
)


def _warm_model() -> None:
    """Load + cache the model at boot, applying the failure policy from 1.5.

    Extracted from the old @app.on_event("startup") body so the same logic can
    be driven by the lifespan handler below.
    """
    from apps.model_loader import load_model

    os.makedirs("models", exist_ok=True)

    try:
        load_model()  # loads models/best_rf_clean.joblib into the cache
    except FileNotFoundError as e:
        hint = (
            "Run the training notebook to produce "
            "models/best_rf_clean.joblib, or point MODEL_PATH at an existing "
            "model file."
        )
        if EAGER_MODEL_LOAD:
            raise RuntimeError(
                f"[startup] FATAL: model file not found ({e}). {hint} "
                "Refusing to start. Set EAGER_MODEL_LOAD=false to boot anyway "
                "(the API will report not-ready until the model appears)."
            ) from e
        logger.error("[startup] Model not available: %s. %s", e, hint)
    except Exception as e:  # corrupt/truncated .joblib, unpickling error, OOM...
        if EAGER_MODEL_LOAD:
            raise RuntimeError(
                f"[startup] FATAL: could not load the model ({e}). Refusing to "
                "start. Set EAGER_MODEL_LOAD=false to boot anyway."
            ) from e
        logger.exception("[startup] Could not load model; continuing not-ready.")
    else:
        logger.info("[startup] Model loaded and cached.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: warm the model before serving, then yield.

    Replaces the deprecated @app.on_event("startup") hook, which FastAPI
    warns about and which has no ordering guarantees against the router.
    """
    _warm_model()
    yield


app = FastAPI(
    title="Smart Grid Load Forecasting API",
    description=(
        "Forecast hourly electrical load (MW) for PJM regions using an "
        "already-trained Random Forest model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS - allow specific origins only (never '*' without a reason).
# ---------------------------------------------------------------------------
# Origin matching is scheme+host+port EXACT, so every port a client can
# come from must be listed. Extra origins can be appended at deploy time via
# ALLOWED_ORIGINS="https://a.com,https://b.com" without editing this file.
origins = [
    "http://localhost",
    "http://127.0.0.1",
    # This FastAPI app itself (run with --port 8001 per frontend/load_forecast.py).
    "http://localhost:8001",
    "http://127.0.0.1:8001",
    # Reflex frontend dev server (frontend/load_forecast.py runs on port 3000).
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

# Append any extra origins from the environment (comma-separated).
_extra = os.getenv("ALLOWED_ORIGINS", "")
origins += [o.strip() for o in _extra.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# Mount the endpoint router under the /api/v1 prefix.
app.include_router(router, prefix="/api/v1")


# (Model warming now happens in the lifespan handler declared above.)


@app.get("/")
def root():
    """Root landing page.

    The frontend in this project is the Reflex app under frontend/, so there
    is no static index.html to serve - this used to branch on a file that
    never existed and silently fall through to the JSON response.
    """
    return {"message": "Smart Grid Load Forecasting API is running"}
