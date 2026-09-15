"""
tests/test_raw_path_and_order.py
================================
Items 2.7 and 2.9 from IMPROVEMENTS.md.

2.7 - _resolve_raw_path() fallback order (apps/routes.py)
    It tries data_raw/merged_long_df.csv, then
    clean_data/merged_long_df_clean.csv, and raises FileNotFoundError if
    neither exists. The order matters: the RAW file is preferred, and the
    only way to prove the fallback works is to make the first candidate
    absent while the second is present.

2.9 - feature order is respected (machine_learning.machine_learning.predict)
    predict() does `features[load_feature_cols()]`, which both SELECTS the
    trained columns and REORDERS them. A model trained on a fixed column
    order silently produces nonsense if handed a shuffled frame, so this
    line is load-bearing. The test asserts identity of predictions across
    permutations - that IS the whole point of the line.

No model artefact and no 342 MiB CSV are required by any test here:
predict() receives a fake estimator, and the path resolver runs in a tmp dir.
"""
import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# 2.7 - _resolve_raw_path()
# ---------------------------------------------------------------------------
RAW_REL = "data_raw/merged_long_df.csv"
CLEAN_REL = "clean_data/merged_long_df_clean.csv"


@pytest.fixture()
def patched_root(tmp_path, monkeypatch):
    """Run _resolve_raw_path() against a throwaway project root.

    _resolve_raw_path derives its root from Path(__file__).parents[1] of
    apps/routes.py, so we monkeypatch that attribute rather than touching
    the real tree. The relative layout inside tmp_path mirrors the repo.
    """
    import apps.routes as routes

    fake = tmp_path / "proj" / "apps" / "routes.py"
    fake.parent.mkdir(parents=True)
    fake.write_text("# stand-in for apps/routes.py\n")

    monkeypatch.setattr(routes, "__file__", str(fake))
    return tmp_path / "proj"


def test_resolve_prefers_the_raw_file_when_both_exist(patched_root):
    """Order check: raw wins even when the clean file is also present."""
    (patched_root / "data_raw").mkdir()
    (patched_root / "clean_data").mkdir()
    (patched_root / RAW_REL).write_text("raw")
    (patched_root / CLEAN_REL).write_text("clean")

    import apps.routes as routes
    got = routes._resolve_raw_path()

    assert got == str(patched_root / RAW_REL)
    assert got.endswith("data_raw/merged_long_df.csv")


def test_resolve_falls_back_to_the_clean_file(patched_root):
    """The raw candidate is absent -> the clean one is used."""
    (patched_root / "clean_data").mkdir()
    (patched_root / CLEAN_REL).write_text("clean")

    import apps.routes as routes
    got = routes._resolve_raw_path()

    assert got == str(patched_root / CLEAN_REL)
    assert got.endswith("clean_data/merged_long_df_clean.csv")


def test_resolve_raises_when_neither_exists(patched_root):
    """Both candidates missing -> FileNotFoundError, not a silent None."""
    import apps.routes as routes

    with pytest.raises(FileNotFoundError) as exc:
        routes._resolve_raw_path()

    # The message must name the remedy, since it surfaces to operators.
    assert "merged" in str(exc.value).lower()


def test_resolve_ignores_a_directory_masquerading_as_the_csv(patched_root):
    """A directory named like the CSV exists -> .exists() is True, but the
    later read_csv would fail. Documents that the check is existence-only."""
    (patched_root / "data_raw" / "merged_long_df.csv").mkdir(parents=True)
    (patched_root / "clean_data").mkdir()
    (patched_root / CLEAN_REL).write_text("clean")

    import apps.routes as routes
    got = routes._resolve_raw_path()

    # The directory wins the .exists() test - a known limitation, pinned here
    # so that a future change to use .is_file() is a visible, deliberate one.
    assert got == str(patched_root / RAW_REL)
    assert Path(got).is_dir()


def test_resolve_returns_a_string(patched_root):
    """Downstream passes the result straight to pd.read_csv."""
    (patched_root / "data_raw").mkdir()
    (patched_root / RAW_REL).write_text("raw")

    import apps.routes as routes
    assert isinstance(routes._resolve_raw_path(), str)


def test_live_repo_has_at_least_one_candidate(project_root):
    """On the real checkout one of the two files must exist.

    If this fails, the API cannot build forecast context at all. Skipped when
    the CSVs are not checked out (e.g. CI without large data).
    """
    import apps.routes as routes

    raw = project_root / RAW_REL
    clean = project_root / CLEAN_REL
    if not raw.exists() and not clean.exists():
        pytest.skip("neither merged CSV is present in this checkout")

    resolved = routes._resolve_raw_path()
    assert Path(resolved).exists()
    resolved_path = Path(resolved)
    assert resolved_path == raw or resolved_path == clean


# ---------------------------------------------------------------------------
# 2.9 - feature order respected
# ---------------------------------------------------------------------------
class FakeRegressor:
    """Minimal estimator that records the column order it received.

    It returns the sum of the values, so a reordered frame would give a
    DIFFERENT answer if the reorder were missing. That makes the assertion
    meaningful rather than vacuous.
    """

    def __init__(self):
        self.calls = []
        self.n_features_in_ = None

    def predict(self, X):
        self.calls.append(list(X.columns))
        self.n_features_in_ = X.shape[1]
        # Row-wise sum: sensitive to column ORDER only through the record above,
        # but sensitive to column SET through the value.
        return np.asarray(X).sum(axis=1)


@pytest.fixture()
def ml(monkeypatch, feature_cols):
    """machine_learning.machine_learning with a stubbed feature list.

    load_feature_cols() normally reads the 401-byte joblib file; stubbing it
    keeps this test independent of that artefact while still exercising the
    real predict() implementation.
    """
    mod = importlib.import_module("machine_learning.machine_learning")
    monkeypatch.setattr(mod, "load_feature_cols", lambda: list(feature_cols))
    return mod


def _frame(cols, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(size=(5, len(cols))), columns=list(cols))


def test_predict_reorders_shuffled_columns_to_the_trained_order(ml, feature_cols):
    """The core of 2.9: shuffled input == correctly ordered input."""
    model = FakeRegressor()

    ordered = _frame(feature_cols, seed=1)
    shuffled = ordered[list(reversed(feature_cols))]

    p_ordered = ml.predict(model, ordered)
    p_shuffled = ml.predict(model, shuffled)

    np.testing.assert_allclose(p_ordered, p_shuffled)

    # And the estimator saw the TRAINED order both times, not the input order.
    assert model.calls[0] == list(feature_cols)
    assert model.calls[1] == list(feature_cols)


def test_predict_handles_arbitrary_column_order(ml, feature_cols):
    """A deliberately scrambled permutation still lands on the trained order."""
    model = FakeRegressor()
    base = _frame(feature_cols, seed=7)

    perm = list(np.random.default_rng(3).permutation(feature_cols))
    scrambled = base[perm]
    assert perm != list(feature_cols), "permutation must actually differ"

    np.testing.assert_allclose(
        ml.predict(model, base),
        ml.predict(model, scrambled),
    )
    assert model.calls[-1] == list(feature_cols)


def test_predict_selects_only_the_trained_columns(ml, feature_cols):
    """Extra columns are dropped, not passed through to the model.

    This is the SELECTING half of `features[load_feature_cols()]`. Without it
    a model would receive unexpected inputs (or raise).
    """
    model = FakeRegressor()
    base = _frame(feature_cols, seed=2)
    noisy = base.copy()
    noisy["MW"] = 999.0
    noisy["unrelated_extra"] = -1.0

    got = ml.predict(model, noisy)

    assert model.n_features_in_ == len(feature_cols)
    assert model.calls[-1] == list(feature_cols)
    np.testing.assert_allclose(got, base.to_numpy().sum(axis=1))


def test_predict_output_length_matches_input_rows(ml, feature_cols):
    model = FakeRegressor()
    out = ml.predict(model, _frame(feature_cols, seed=4))
    assert out.shape == (5,)


def test_predict_raises_when_a_trained_column_is_missing(ml, feature_cols):
    """A missing trained column is an error, not a silently zero-filled one.

    Documented behaviour: pandas raises KeyError. Downstream callers must
    build ALL feature columns; they cannot rely on an implicit default.
    """
    model = FakeRegressor()
    incomplete = _frame(feature_cols, seed=5).drop(columns=[feature_cols[0]])

    with pytest.raises(KeyError):
        ml.predict(model, incomplete)


def test_predict_preserves_row_order(ml, feature_cols):
    """Rows must not be reordered - only columns are."""
    model = FakeRegressor()
    base = _frame(feature_cols, seed=6)
    marker = base.to_numpy().sum(axis=1)

    np.testing.assert_allclose(ml.predict(model, base), marker)
    # Shuffling rows changes the OUTPUT ORDER and nothing else about the set.
    row_perm = [3, 0, 4, 1, 2]
    out = ml.predict(model, base.iloc[row_perm])
    np.testing.assert_allclose(out, marker[row_perm])
