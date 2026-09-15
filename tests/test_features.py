"""
tests/test_features.py
======================
Tests for `_build_future_row()` (IMPROVEMENTS.md item 2.4).

This is the highest-value test file in the repo: it covers pure, zero-I/O
logic that feeds every single prediction, and it is the only thing standing
between a silent indexing bug and confidently wrong megawatts.

It also guards the Phase-1 rewrite (item 1.11): the running load series moved
from `np.append` inside the loop to a pre-allocated buffer, so the lag and
rolling features must still read exactly the same values.
"""
import numpy as np
import pandas as pd
import pytest

from apps.routes import _build_future_row


# ---------------------------------------------------------------------------
# 2.4f - the dict must contain EXACTLY the trained feature set
# ---------------------------------------------------------------------------
def test_returns_exactly_the_26_trained_features(feature_cols):
    ts = pd.Timestamp("2018-06-15 14:00:00")
    y = np.arange(1000, 2000, dtype="float64")  # plenty of history
    row = _build_future_row(ts, y, feature_cols)

    assert set(row) == set(feature_cols)
    assert len(row) == 26


def test_no_extra_keys_beyond_feature_list(feature_cols):
    """A stray key would silently produce a wrong-order DataFrame later."""
    ts = pd.Timestamp("2018-06-15 14:00:00")
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(ts, y, feature_cols)
    assert not (set(row) - set(feature_cols))


# ---------------------------------------------------------------------------
# 2.4a - cyclical hour encoding
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("hour", [0, 6, 12, 14, 18, 23])
def test_hour_sin_cos_match_the_formula(feature_cols, hour):
    ts = pd.Timestamp("2018-06-15") + pd.Timedelta(hours=hour)
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(ts, y, feature_cols)

    assert row["hour"] == hour
    assert row["hour_sin"] == pytest.approx(np.sin(2 * np.pi * hour / 24))
    assert row["hour_cos"] == pytest.approx(np.cos(2 * np.pi * hour / 24))


@pytest.mark.parametrize("month", [1, 3, 6, 9, 12])
def test_month_sin_cos_match_the_formula(feature_cols, month):
    ts = pd.Timestamp(year=2018, month=month, day=15, hour=12)
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(ts, y, feature_cols)

    assert row["month"] == month
    assert row["month_sin"] == pytest.approx(np.sin(2 * np.pi * month / 12))
    assert row["month_cos"] == pytest.approx(np.cos(2 * np.pi * month / 12))


@pytest.mark.parametrize("weekday", range(7))
def test_day_of_week_sin_cos_match_the_formula(feature_cols, weekday):
    # 2018-06-11 is a Monday (weekday 0).
    ts = pd.Timestamp("2018-06-11") + pd.Timedelta(days=weekday) + pd.Timedelta(hours=9)
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(ts, y, feature_cols)

    assert row["day_of_week"] == weekday
    assert row["day_of_week_sin"] == pytest.approx(np.sin(2 * np.pi * weekday / 7))
    assert row["day_of_week_cos"] == pytest.approx(np.cos(2 * np.pi * weekday / 7))


@pytest.mark.parametrize("hour", [0, 1, 5, 12, 23])
def test_fourier_terms_are_shifted_by_one_hour(feature_cols, hour):
    """fourier_* uses (h - 1), not h - assert the shift explicitly."""
    ts = pd.Timestamp("2018-06-15") + pd.Timedelta(hours=hour)
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(ts, y, feature_cols)

    assert row["fourier_sin_1"] == pytest.approx(np.sin(2 * np.pi * (hour - 1) / 24))
    assert row["fourier_cos_1"] == pytest.approx(np.cos(2 * np.pi * (hour - 1) / 24))
    assert row["fourier_sin_2"] == pytest.approx(np.sin(2 * np.pi * (hour - 1) / 12))
    assert row["fourier_cos_2"] == pytest.approx(np.cos(2 * np.pi * (hour - 1) / 12))


# ---------------------------------------------------------------------------
# 2.4b - is_weekend
# ---------------------------------------------------------------------------
def test_is_weekend_is_1_on_saturday_and_sunday(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    # 2018-06-16 = Saturday, 2018-06-17 = Sunday
    for day in ("2018-06-16", "2018-06-17"):
        row = _build_future_row(pd.Timestamp(day + " 12:00"), y, feature_cols)
        assert row["is_weekend"] == 1, day


def test_is_weekend_is_0_on_a_wednesday(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-13 12:00"), y, feature_cols)
    assert row["is_weekend"] == 0


# ---------------------------------------------------------------------------
# 2.4c - lag features read exact positions off the running series
# ---------------------------------------------------------------------------
def test_lag_1h_reads_the_last_value(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert row["load_lag_1h"] == y[-1]


def test_lag_24h_reads_exactly_y_minus_24(feature_cols):
    """The single most important lag: same hour, previous day."""
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert row["load_lag_24h"] == y[-24]


def test_lag_168h_reads_exactly_y_minus_168(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert row["load_lag_168h"] == y[-168]


def test_lag_720h_reads_exactly_y_minus_720(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert row["load_lag_720h"] == y[-720]


def test_lags_are_distinct_so_an_off_by_one_is_visible(feature_cols):
    """If a lag index were wrong two features would collide."""
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert len({row["load_lag_1h"], row["load_lag_24h"],
                row["load_lag_168h"], row["load_lag_720h"]}) == 4


# ---------------------------------------------------------------------------
# 2.4d - rolling windows
# ---------------------------------------------------------------------------
def test_roll_mean_24h_equals_last_24_mean(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert row["load_roll_mean_24h"] == pytest.approx(y[-24:].mean())


def test_roll_std_24h_equals_last_24_std(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert row["load_roll_std_24h"] == pytest.approx(y[-24:].std())


def test_roll_mean_168h_equals_last_168_mean(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert row["load_roll_mean_168h"] == pytest.approx(y[-168:].mean())


def test_roll_std_168h_equals_last_168_std(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert row["load_roll_std_168h"] == pytest.approx(y[-168:].std())


def test_rolling_windows_use_only_the_tail(feature_cols):
    """Change a value OUTSIDE the 24h window; the 24h mean must not move."""
    base = np.arange(1000, 2000, dtype="float64")
    ts = pd.Timestamp("2018-06-15 14:00")

    row_a = _build_future_row(ts, base.copy(), feature_cols)

    perturbed = base.copy()
    perturbed[0] += 99999.0  # far outside the last 24 points
    row_b = _build_future_row(ts, perturbed, feature_cols)

    assert row_a["load_roll_mean_24h"] == pytest.approx(row_b["load_roll_mean_24h"])
    assert row_a["load_lag_24h"] == row_b["load_lag_24h"]


# ---------------------------------------------------------------------------
# 2.4e - short history must yield NaN, never raise
# ---------------------------------------------------------------------------
def test_short_history_yields_nan_for_720h_lag(feature_cols):
    """Only 10 points of history: 720h lag cannot exist -> NaN, not crash."""
    y = np.arange(10, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)

    assert np.isnan(row["load_lag_720h"])
    assert np.isnan(row["load_lag_168h"])
    assert np.isnan(row["load_lag_24h"])
    # 1h lag DOES exist with 10 points.
    assert row["load_lag_1h"] == y[-1]


def test_short_history_still_produces_every_key(feature_cols):
    y = np.arange(10, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert set(row) == set(feature_cols)


def test_exactly_24_points_allows_the_24h_lag(feature_cols):
    """Boundary: len(y) >= 24 is the condition, so 24 exactly must work."""
    y = np.arange(24, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert row["load_lag_24h"] == y[-24]
    assert not np.isnan(row["load_lag_24h"])
    # ...but 168h still cannot be satisfied.
    assert np.isnan(row["load_lag_168h"])


def test_23_points_is_not_enough_for_the_24h_lag(feature_cols):
    """Boundary the other way: 23 points -> NaN."""
    y = np.arange(23, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    assert np.isnan(row["load_lag_24h"])


# ---------------------------------------------------------------------------
# 2.4 (extra) - guards for the Phase-1 pre-allocated-buffer rewrite (item 1.11)
# ---------------------------------------------------------------------------
def test_a_view_of_a_slice_behaves_like_a_standalone_array(feature_cols):
    """_recursive_forecast now passes `y[:n_hist + step]`, a VIEW.

    Indexing a view must give identical results to indexing a fresh array,
    otherwise the lag features would silently change with the buffer rewrite.
    """
    ts = pd.Timestamp("2018-06-15 14:00")
    full = np.arange(1000, 2000, dtype="float64")

    row_full = _build_future_row(ts, full, feature_cols)
    row_view = _build_future_row(ts, full[: len(full)], feature_cols)

    assert row_full == row_view


def test_row_is_json_serialisable(feature_cols):
    """Every value must be a plain int/float so it can go into a DataFrame
    and then out through the API."""
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 14:00"), y, feature_cols)
    for key, value in row.items():
        assert isinstance(value, (int, float, np.integer, np.floating)), key


def test_calendar_flags(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")

    first = _build_future_row(pd.Timestamp("2018-06-01 12:00"), y, feature_cols)
    assert first["is_month_start"] == 1
    assert first["is_month_end"] == 0

    last = _build_future_row(pd.Timestamp("2018-06-30 12:00"), y, feature_cols)
    assert last["is_month_end"] == 1
    assert last["is_month_start"] == 0

    mid = _build_future_row(pd.Timestamp("2018-06-15 12:00"), y, feature_cols)
    assert mid["is_month_start"] == 0
    assert mid["is_month_end"] == 0


def test_year_is_taken_from_the_timestamp(feature_cols):
    y = np.arange(1000, 2000, dtype="float64")
    row = _build_future_row(pd.Timestamp("2018-06-15 12:00"), y, feature_cols)
    assert row["year"] == 2018


def test_is_holiday_is_always_zero(feature_cols):
    """Documented behaviour: the training notebook also used is_holiday = 0
    ('no holidays in this dataset'), so inference matching it is CORRECT.
    If this ever changes, the model must be retrained - hence this test."""
    y = np.arange(1000, 2000, dtype="float64")

    # 2018-07-04 is US Independence Day, 2018-12-25 is Christmas: real
    # holidays, and the model still receives 0 for both. That is the
    # documented, training-consistent behaviour (see book2_loaded.ipynb).
    for day in ("2018-07-04", "2018-12-25"):
        row = _build_future_row(pd.Timestamp(day + " 12:00"), y, feature_cols)
        assert row["is_holiday"] == 0, day

    # And a non-holiday day is also 0 - i.e. the feature is constant.
    normal = _build_future_row(pd.Timestamp("2018-06-15 12:00"), y, feature_cols)
    assert normal["is_holiday"] == 0
