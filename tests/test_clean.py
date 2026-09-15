"""
tests/test_clean.py
===================
Tests for `scripts/clean.py::_clean_chunk()` (IMPROVEMENTS.md item 2.5).

`_clean_chunk` takes a plain DataFrame and returns a cleaned one - no file I/O,
no model. So every test here builds a tiny frame in memory and asserts on the
result. Fast, offline, and it pins the cleaning contract that the training
data depends on.
"""
import numpy as np
import pandas as pd
import pytest

from scripts.clean import KEEP_COLS, _clean_chunk, target_col

LAG_COLS = [
    "load_lag_1h", "load_lag_24h", "load_lag_168h", "load_lag_720h",
    "load_roll_mean_24h", "load_roll_std_24h", "load_roll_mean_168h",
    "load_roll_std_168h",
]


def _row(**overrides) -> dict:
    """A complete, valid row - every KEEP_COL present and non-null."""
    row = {c: 1.0 for c in KEEP_COLS}
    row["Datetime"] = "2018-06-15 14:00:00"
    row["region"] = "AEP"
    row["MW"] = 12345.0
    row.update(overrides)
    return row


def _frame(n: int = 3, **overrides) -> pd.DataFrame:
    """A small frame of `n` valid rows, with overrides applied to all rows."""
    return pd.DataFrame([_row(**overrides) for _ in range(n)])


# ---------------------------------------------------------------------------
# Core contract: rows survive, columns are trimmed to KEEP_COLS
# ---------------------------------------------------------------------------
def test_valid_rows_survive_untouched():
    df = _frame(5)
    out = _clean_chunk(df)
    assert len(out) == 5


def test_output_columns_are_exactly_the_keep_cols_that_were_present():
    """Extra input columns are dropped; nothing new is invented."""
    df = _frame(2)
    df["not_a_feature"] = 99
    df["another_junk_col"] = "x"

    out = _clean_chunk(df)

    assert set(out.columns) == set(KEEP_COLS)
    assert "not_a_feature" not in out.columns
    assert "another_junk_col" not in out.columns


def test_output_column_order_follows_keep_cols():
    """Order matters downstream - it must follow KEEP_COLS, not the input."""
    df = _frame(2)
    df = df[[c for c in reversed(list(df.columns))]]  # shuffle the input

    out = _clean_chunk(df)

    expected = [c for c in KEEP_COLS if c in out.columns]
    assert list(out.columns) == expected


def test_missing_keep_cols_are_not_created():
    """If the input lacks a KEEP_COL, it is simply absent - not fabricated."""
    df = _frame(2).drop(columns=["fourier_sin_1", "quarter"])
    out = _clean_chunk(df)

    assert "fourier_sin_1" not in out.columns
    assert "quarter" not in out.columns
    assert set(out.columns) == set(KEEP_COLS) - {"fourier_sin_1", "quarter"}


# ---------------------------------------------------------------------------
# Step 2: rows with no target (MW) are dropped
# ---------------------------------------------------------------------------
def test_nan_mw_rows_are_dropped():
    df = pd.DataFrame([
        _row(MW=1000.0),
        _row(MW=np.nan),
        _row(MW=3000.0),
    ])
    out = _clean_chunk(df)

    assert len(out) == 2
    assert out["MW"].tolist() == [1000.0, 3000.0]


def test_all_nan_mw_returns_empty_frame():
    df = pd.DataFrame([_row(MW=np.nan) for _ in range(3)])
    out = _clean_chunk(df)
    assert len(out) == 0


def test_none_among_valid_mw_is_dropped():
    """None counts as missing, same as NaN."""
    df = pd.DataFrame([_row(MW=1000.0), _row(MW=None)])
    out = _clean_chunk(df)
    assert len(out) == 1


def test_zero_mw_is_kept():
    """0 is a legitimate value - dropna must not confuse it with missing."""
    df = pd.DataFrame([_row(MW=0.0)])
    out = _clean_chunk(df)
    assert len(out) == 1
    assert out["MW"].iloc[0] == 0.0


def test_negative_mw_survives_cleaning():
    """Cleaning does not judge values - it only drops missing ones."""
    df = pd.DataFrame([_row(MW=-500.0)])
    out = _clean_chunk(df)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Step 3: NaN lag / rolling rows are dropped (the first hours of a region)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("lag_col", LAG_COLS)
def test_row_with_any_nan_lag_is_dropped(lag_col):
    """Each lag column individually is enough to disqualify a row."""
    df = pd.DataFrame([_row(), _row(**{lag_col: np.nan}), _row()])
    out = _clean_chunk(df)

    assert len(out) == 2, f"row with NaN {lag_col} should have been dropped"


def test_row_kept_when_all_lags_present():
    df = _frame(1)
    out = _clean_chunk(df)
    assert len(out) == 1


def test_lag_nan_and_mw_nan_are_both_dropped():
    df = pd.DataFrame([
        _row(),                                  # kept
        _row(MW=np.nan),                         # dropped (target)
        _row(load_lag_720h=np.nan),              # dropped (lag)
        _row(),                                  # kept
    ])
    out = _clean_chunk(df)
    assert len(out) == 2


def test_absent_lag_column_does_not_drop_everything():
    """The dropna subset is filtered to columns that exist.

    If it were not, a frame without lag columns would be wiped out entirely.
    """
    df = _frame(3).drop(columns=LAG_COLS)
    out = _clean_chunk(df)
    assert len(out) == 3


# ---------------------------------------------------------------------------
# Step 4: numeric coercion safety net
# ---------------------------------------------------------------------------
def test_numeric_strings_are_coerced_to_numbers():
    """Values arriving as text must become numbers."""
    df = _frame(1)
    df["MW"] = "12345.0"
    df["hour"] = "14"

    out = _clean_chunk(df)

    assert pd.api.types.is_numeric_dtype(out["MW"])
    assert out["MW"].iloc[0] == 12345.0
    assert out["hour"].iloc[0] == 14


def test_coercion_failure_in_lag_survives_dropna_then_becomes_nan():
    """Documents an ORDERING subtlety in _clean_chunk.

    The dropna filters (steps 2-3) run BEFORE numeric coercion (step 4). So a
    value that is a non-numeric STRING is not "missing" at dropna time and
    the row survives - only afterwards does coercion turn it into NaN.

    In production this is masked by na_values= in read_csv, which converts the
    real sentinels up front. This test pins the actual behaviour so that if
    the ordering is ever changed, the change is deliberate and visible.
    """
    df = _frame(1, load_lag_24h="not-a-number")
    out = _clean_chunk(df)

    assert len(out) == 1                          # survived dropna
    assert pd.isna(out["load_lag_24h"].iloc[0])   # then coerced to NaN


def test_unparseable_mw_survives_dropna_then_becomes_nan():
    """Same ordering effect on the target column itself.

    'garbage' is not null at step 2, so the row is kept; step 4 then coerces
    MW to NaN. The result is a row with a NaN target - which downstream code
    would have to handle, or which the reader's na_values should have caught.
    """
    df = _frame(1)
    df["MW"] = "garbage"
    out = _clean_chunk(df)

    assert len(out) == 1
    assert pd.isna(out["MW"].iloc[0])


def test_text_sentinels_are_handled_by_the_reader_not_the_cleaner():
    """The safe path: na_values converts sentinels to NaN BEFORE cleaning.

    With text already NaN on arrival, the step-2 dropna works as intended and
    the row is removed. This is why production data is fine and only malformed
    input would expose the ordering gap above.
    """
    df = _frame(1)
    df["MW"] = np.nan   # what na_values would have produced
    out = _clean_chunk(df)

    assert len(out) == 0


def test_datetime_and_region_are_left_alone():
    """These two are explicitly excluded from numeric coercion."""
    df = _frame(1)
    out = _clean_chunk(df)

    assert out["region"].iloc[0] == "AEP"
    assert not pd.api.types.is_numeric_dtype(out["region"])
    # Datetime must still parse as a date, not have become a number.
    assert pd.to_datetime(out["Datetime"].iloc[0]) == pd.Timestamp("2018-06-15 14:00")


# ---------------------------------------------------------------------------
# Sentinels: -9 / -9.0 / '?' are read as missing by the CSV reader
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("sentinel", ["-9", "-9.0", "?"])
def test_na_values_sentinels_are_read_as_missing(tmp_path, sentinel):
    """The reader is configured with na_values=["-9","-9.0","?"].

    This drives the real pd.read_csv call with that setting, so the sentinel
    handling is verified through the same path production uses - not by
    hand-placing a NaN.
    """
    csv = tmp_path / "sentinel.csv"
    header = ",".join(KEEP_COLS)
    good = ",".join(str(_row()[c]) for c in KEEP_COLS)
    rows = [header, good, good.replace("12345.0", sentinel, 1)]
    csv.write_text("\n".join(rows) + "\n")

    read = pd.read_csv(
        csv,
        usecols=KEEP_COLS,
        na_values=["-9", "-9.0", "?"],
    )

    # The sentinel row must have arrived as NaN in the MW column...
    assert read["MW"].isna().sum() == 1, f"{sentinel} should map to NaN"

    # ...and therefore be dropped by _clean_chunk.
    out = _clean_chunk(read)
    assert len(out) == 1


def test_a_real_minus_9_mw_survives_without_na_values(tmp_path):
    """Guard the guard: -9 IS a valid megawatt value.

    The sentinel list is what makes it missing. Without na_values it must be
    kept - proving the behaviour comes from that setting and not from the
    cleaning logic itself.
    """
    csv = tmp_path / "minus9.csv"
    header = ",".join(KEEP_COLS)
    good = ",".join(str(_row()[c]) for c in KEEP_COLS)
    bad = good.replace("12345.0", "-9", 1)
    csv.write_text("\n".join([header, bad]) + "\n")

    read = pd.read_csv(csv, usecols=KEEP_COLS)  # no na_values
    assert read["MW"].iloc[0] == -9.0

    out = _clean_chunk(read)
    assert len(out) == 1


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------
def test_missing_mw_column_does_not_raise():
    """Regression guard for the KeyError found while writing these tests.

    Before the fix, dropna(subset=['MW']) ran even when MW was absent and
    raised KeyError - killing the entire cleaning run on the first chunk.
    A missing target column must degrade to 'nothing to clean', not crash.
    """
    df = _frame(2).drop(columns=["MW"])
    out = _clean_chunk(df)  # must not raise
    assert "MW" not in out.columns
    assert len(out.columns) > 0


def test_frame_with_mw_but_no_other_features_keeps_rows():
    """MW alone is enough to survive step 2 (no lag columns present)."""
    out = _clean_chunk(pd.DataFrame({"MW": [1.0, 2.0, np.nan]}))
    assert len(out) == 2


def test_frame_with_no_keep_cols_returns_no_columns():
    """A frame sharing no columns with KEEP_COLS comes back column-less.

    Note we assert on the COLUMNS, not len(): the result legitimately keeps
    its row index even though it has no columns, so len() would be 3 here.
    That is exactly why this test checks .columns instead.
    """
    out = _clean_chunk(pd.DataFrame({"unrelated": [1, 2, 3]}))
    assert len(out.columns) == 0


def test_does_not_mutate_the_input_frame():
    """Cleaning must not have side effects on the caller's frame."""
    df = _frame(2)
    df["junk"] = 1
    before_cols = list(df.columns)
    before_len = len(df)

    _clean_chunk(df)

    assert list(df.columns) == before_cols
    assert len(df) == before_len
