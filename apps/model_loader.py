"""
model_loader.py
===============
Load the trained load-forecasting model ONCE and cache it in memory.

Mirrors the heart-disease project's model_loader, but points at the saved
RandomForestRegressor (models/best_rf_clean.joblib).

Why cache?
  The model file is large (~1 GB). Loading it on every request would be slow
  and waste memory. Instead we load it the first time it is needed and keep
  the object in a module-level variable so later requests reuse it instantly.
"""
import os
import joblib
from pathlib import Path

# Module-level cache: None until the model is loaded the first time.
_model = None

# Default to this project's saved model (override with MODEL_PATH env var).
DEFAULT_MODEL_PATH = str(
    Path(__file__).resolve().parents[1] / "models" / "best_rf_clean.joblib"
)


def _resolve_path() -> str:
    """Return the model path from env var, falling back to the default."""
    path = os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not path:
        path = DEFAULT_MODEL_PATH
    return path


def load_model():
    """Load the model from disk into the module cache.

    Returns the RandomForestRegressor object directly (the .joblib holds the
    bare model - not a dict, unlike the heart-disease version).
    """
    global _model
    path = _resolve_path()
    print(f"[model_loader] Loading model from: {path}")
    _model = joblib.load(path)
    return _model


def get_model():
    """Return the cached model, loading it from disk if not yet loaded."""
    global _model
    if _model is None:
        load_model()
    return _model


def model_loaded() -> bool:
    """Convenience: is the model already in memory?"""
    return _model is not None
