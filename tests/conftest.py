"""
Shared pytest fixtures for the SMART_GRID_LOAD_FORECASTING test suite.

Design rule for this suite (see IMPROVEMENTS.md Phase 2):
    NOTHING here may require the 1.05 GiB model artefact or the 342 MiB CSV.
    Every test must run in seconds on CI with no large downloads.

Two escape hatches make that possible:
  * `feature_cols` reads the 401-byte feature-name list (tiny, committed).
  * `fake_model` is a stand-in regressor, so the API can be exercised without
    unpickling a gigabyte of RandomForest.
"""
import sys
from pathlib import Path

import pytest

# Make the project importable as `apps.*` / `scripts.*` regardless of the
# directory pytest was invoked from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

FEATURE_COLS_PATH = PROJECT_ROOT / "models" / "feature_cols_clean.joblib"


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the repository root."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def feature_cols() -> list[str]:
    """The exact 26 feature names the model was trained on.

    Read from the committed 401-byte joblib - NOT from the big model file.
    """
    import joblib

    if not FEATURE_COLS_PATH.exists():
        pytest.skip(f"feature list not found at {FEATURE_COLS_PATH}")
    cols = list(joblib.load(FEATURE_COLS_PATH))
    assert len(cols) == 26, f"expected 26 features, got {len(cols)}"
    return cols


class FakeRegressor:
    """Minimal sklearn-like regressor for API tests.

    `predict` must accept a DataFrame and return an array-like of the same
    length. The value returned is deliberately deterministic and depends only
    on the row count, so tests can assert exact numbers without a real model.
    """

    def __init__(self, value: float = 12345.0):
        self.value = value
        self.calls: list[list[str]] = []  # column order seen per call

    def predict(self, X):  # noqa: N803 - sklearn naming
        import numpy as np

        # Record the column order so tests can prove reindexing happened.
        self.calls.append(list(getattr(X, "columns", [])))
        n = len(X)
        return np.full(n, self.value, dtype="float64")


@pytest.fixture
def fake_model() -> FakeRegressor:
    """A 12_345.0-returning stand-in for the Random Forest."""
    return FakeRegressor()
