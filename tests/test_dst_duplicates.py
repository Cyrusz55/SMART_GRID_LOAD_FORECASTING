"""
tests/test_dst_duplicates.py
============================
Regression tests for the DST fall-back duplication (IMPROVEMENTS.md item 2.6).

THE BUG (found in notebooks/book1.ipynb, cell 4)
------------------------------------------------
The original pipeline built the training frame with a WIDE merge on Datetime:

    pd.merge(left, right, on="Datetime")     # 12 times, once per region

A DST fall-back repeats the 02:00 wall-clock hour, so a region's file holds
two rows for that timestamp. Merging two frames that each have a duplicate key
produces the CROSS PRODUCT: 2 x 2 = 4 rows. Chaining 10 such merges gave
2**10 = 1024 rows at each fall-back hour. Two distinct failure modes:

  * how="outer" -> row explosion (2**10 = 1024 rows at the repeated hour)
  * how="inner" -> EMPTY frame, because the regions cover different date
                   ranges so no timestamp exists in all 12 files

The fix was to stop merging entirely and build LONG form (one row per
region+timestamp) with pd.concat, plus a per-file
`drop_duplicates(subset="Datetime")` BEFORE any shift/lag logic.

WHY THE DEDUP ORDER MATTERS
---------------------------
book1 says it directly: "Keep just the first one so each timestamp is unique
BEFORE any shift/lag logic (otherwise lags point at fake 'previous' hours)."

A duplicated hour makes load_lag_24h point at a fake neighbour. So the row
count is only the visible symptom - the real hazard is silently wrong lags,
exactly what tests/test_features.py guards from the other direction.

These tests need no real data and no model: the duplication is CONSTRUCTED.
"""
import pandas as pd
import pytest

# 2018-11-04 01:00 US/Eastern occurred twice (fall back: 02:00 EDT -> 01:00 EST).
# 2018-03-11 02:00 did not occur at all (spring forward).
FALL_BACK_HOUR = pd.Timestamp("2018-11-04 01:00:00")
SPRING_FORWARD_HOUR = pd.Timestamp("2018-03-11 02:00:00")


def _region_frame(region: str, stamps) -> pd.DataFrame:
    """One region's long-form slice, with MW values that encode position.

    The value is the ROW's index so a wrong neighbour is detectable - not just
    a wrong count. This is what makes the lag assertion meaningful.
    """
    return pd.DataFrame({
        "Datetime": list(stamps),
        "region": region,
        "MW": [float(1000 + i) for i in range(len(list(stamps)))],
    })


def _stamps_with_fall_back_dupe(periods: int = 240) -> list:
    """Hourly stamps ending at the fall-back hour, which appears TWICE."""
    stamps = list(pd.date_range(end=FALL_BACK_HOUR, periods=periods, freq="h"))
    stamps.append(FALL_BACK_HOUR)  # the repeated wall-clock hour
    return stamps


# ---------------------------------------------------------------------------
# The mechanism: a wide merge on a duplicated key is a cross product
# ---------------------------------------------------------------------------
def test_merge_on_duplicated_key_produces_a_cross_product():
    """This IS the bug, reproduced. 2 x 2 = 4 rows for one timestamp."""
    a = pd.DataFrame({"Datetime": [FALL_BACK_HOUR, FALL_BACK_HOUR], "A": [1, 2]})
    b = pd.DataFrame({"Datetime": [FALL_BACK_HOUR, FALL_BACK_HOUR], "B": [3, 4]})

    merged = pd.merge(a, b, on="Datetime")

    assert len(merged) == 4, "merge on a duplicated key must fan out"
    assert merged["Datetime"].nunique() == 1


def test_ten_chained_merges_give_1024_rows():
    """2**10 - the number quoted in the book1 comment.

    IMPORTANT: this chains merges between DISTINCT frames (one per region),
    which is what the original pipeline did. A self-merge (frame with itself)
    is NOT the same thing - pandas materialises an M x M join index and blows
    up on memory (that mistake cost this test one 32 GiB allocation attempt).
    """
    # Ten separate frames, each holding the duplicated fall-back hour.
    frames = [
        pd.DataFrame({"Datetime": [FALL_BACK_HOUR, FALL_BACK_HOUR], f"r{i}": [1, 2]})
        for i in range(10)
    ]

    merged = frames[0]
    for nxt in frames[1:]:
        merged = pd.merge(merged, nxt, on="Datetime")

    assert len(merged) == 2 ** 10 == 1024
    assert merged["Datetime"].nunique() == 1


def test_chained_merge_row_count_doubles_each_time():
    """Show the doubling explicitly, one step at a time.

    Starting from a single duplicated timestamp, each additional region frame
    doubles the row count: 2, 4, 8, ... 1024. This is the mechanism, and it is
    why the explosion is invisible for the first couple of joins and lethal by
    the tenth.
    """
    rows = 2
    merged = pd.DataFrame({"Datetime": [FALL_BACK_HOUR, FALL_BACK_HOUR]})
    for i in range(1, 10):
        nxt = pd.DataFrame({"Datetime": [FALL_BACK_HOUR, FALL_BACK_HOUR], f"r{i}": [1, 2]})
        merged = pd.merge(merged, nxt, on="Datetime")
        rows *= 2
        assert len(merged) == rows

    assert len(merged) == 1024


def test_inner_merge_of_disjoint_regions_is_empty():
    """The OTHER failure mode: inner merge across regions with no overlap."""
    a = pd.DataFrame({
        "Datetime": pd.date_range("1998-01-01", periods=5, freq="h"),
        "AEP": range(5),
    })
    b = pd.DataFrame({
        "Datetime": pd.date_range("2010-01-01", periods=5, freq="h"),
        "COMED": range(5),
    })

    assert len(pd.merge(a, b, on="Datetime", how="inner")) == 0
    # ...while an outer merge would have exploded instead.
    assert len(pd.merge(a, b, on="Datetime", how="outer")) > 0


# ---------------------------------------------------------------------------
# The FIX: long form + per-file dedup
# ---------------------------------------------------------------------------
def test_long_form_concat_is_immune_to_the_cross_product():
    """12 regions each with a duplicated hour -> 12 extra rows, NOT 2**12.

    concat simply stacks. That is the whole reason the fix works.

    NOTE on the uniqueness contract: in LONG form the same Datetime appears
    once per region, so duplicated(Datetime) is large and CORRECT. The key
    that must be unique is the COMPOSITE (region, Datetime) - which is exactly
    why the notebooks use subset=[SERIES_ID, DATE_COL].
    """
    n_regions, n_stamps = 12, 100
    stamps = list(pd.date_range(end=FALL_BACK_HOUR, periods=n_stamps - 1, freq="h"))
    stamps.append(FALL_BACK_HOUR)  # duplicate inside the region's own stamps

    frames = [_region_frame(f"R{i}", stamps) for i in range(n_regions)]
    long_df = pd.concat(frames, ignore_index=True)

    # Each region contributes its rows independently: no fan-out.
    assert len(long_df) == n_regions * n_stamps

    # Datetime alone is heavily duplicated: every timestamp appears once per
    # region, so only the n_stamps DISTINCT labels are first-occurrences and
    # everything else (including each region's own DST dupe) counts as a
    # repeat. Deriving the number instead of hardcoding it keeps this honest.
    expected_datetime_dupes = n_regions * n_stamps - len(set(stamps))
    assert long_df["Datetime"].duplicated().sum() == expected_datetime_dupes

    # The duplicate INSIDE each region is what must be removed: 12 of them.
    within_region_dupes = long_df.duplicated(subset=["region", "Datetime"]).sum()
    assert within_region_dupes == n_regions

    # After the per-region dedup the composite key is unique...
    deduped = long_df.drop_duplicates(subset=["region", "Datetime"])
    assert deduped.duplicated(subset=["region", "Datetime"]).sum() == 0

    # ...while plain Datetime is still legitimately repeated across regions.
    assert deduped["Datetime"].duplicated().sum() > 0


def test_composite_key_is_the_right_uniqueness_contract():
    """A concrete demo that Datetime-only uniqueness is the WRONG check."""
    long_df = pd.concat([
        _region_frame("AEP", pd.date_range(end=FALL_BACK_HOUR, periods=5, freq="h")),
        _region_frame("COMED", pd.date_range(end=FALL_BACK_HOUR, periods=5, freq="h")),
    ], ignore_index=True)

    # Same timestamps in both regions: 'duplicated' is True almost everywhere.
    assert long_df["Datetime"].duplicated().sum() == 5
    # But no (region, Datetime) pair repeats twice.
    assert long_df.duplicated(subset=["region", "Datetime"]).sum() == 0


def test_dedup_per_file_makes_each_timestamp_unique():
    """drop_duplicates(subset='Datetime') on a single region's frame."""
    stamps = _stamps_with_fall_back_dupe(periods=50)
    df = _region_frame("AEP", stamps)

    assert df["Datetime"].duplicated().sum() == 1  # the duplicate is there

    deduped = df.drop_duplicates(subset="Datetime")

    assert len(deduped) == 50
    assert deduped["Datetime"].is_unique
    # keep="first" is the documented behaviour: the FIRST occurrence survives.
    assert deduped["MW"].iloc[-1] == df["MW"].iloc[49]


def test_dedup_removes_exactly_the_dst_extra_rows():
    """Row count after dedup equals the number of DISTINCT timestamps."""
    stamps = _stamps_with_fall_back_dupe(periods=240)
    df = _region_frame("AEP", stamps)

    before = len(df)
    after = len(df.drop_duplicates(subset="Datetime"))

    assert before - after == 1
    assert after == df["Datetime"].nunique()


# ---------------------------------------------------------------------------
# The property that actually matters: lags must not point at fake hours
# ---------------------------------------------------------------------------
def test_duplicate_hour_makes_lag_24h_point_at_the_wrong_value():
    """Demonstrates the hazard book1's comment warns about."""
    stamps = _stamps_with_fall_back_dupe(periods=50)
    polluted = _region_frame("AEP", stamps)          # has the extra hour
    clean = polluted.drop_duplicates(subset="Datetime")

    # lag_24h at the end reads 24 positions back. With the duplicate present,
    # that lands on a different row than it does in the deduped series.
    polluted_lag = polluted["MW"].iloc[-24]
    clean_lag = clean["MW"].iloc[-24]

    assert polluted_lag != clean_lag, "the duplicate must shift the lag window"

    # And the deduped series is the correct, continuous one.
    assert clean["Datetime"].is_unique
    assert len(clean) == clean["Datetime"].nunique()


def test_dedup_before_dropping_nan_is_the_safe_order():
    """Dedup must come FIRST, before any dropna.

    If a NaN were dropped first, the duplicate could be the surviving row and
    the deletion would silently change which value the lags see.
    """
    stamps = _stamps_with_fall_back_dupe(periods=50)
    df = _region_frame("AEP", stamps)
    df.loc[df.index[-1], "MW"] = float("nan")  # the LATER duplicate is NaN

    # Correct order: dedup first (keeps the first, non-NaN one).
    safe = df.drop_duplicates(subset="Datetime").dropna(subset=["MW"])

    assert len(safe) == 50
    assert safe["MW"].notna().all()
    assert safe["Datetime"].is_unique


# ---------------------------------------------------------------------------
# What the real data on disk looks like (guards against silent regression)
# ---------------------------------------------------------------------------
def test_raw_csvs_still_contain_the_dst_duplicates(project_root):
    """The 40 duplicates are REAL and still on disk.

    This documents that data_raw/ has NOT been pre-deduped, which is why
    book1's per-file drop_duplicates is load-bearing rather than cosmetic.
    Skipped if the raw files are absent (e.g. a CI checkout without data).
    """
    raw = sorted((project_root / "data_raw").glob("*_hourly.csv"))
    if not raw:
        pytest.skip("data_raw/*_hourly.csv not present")

    total_dupes = 0
    files_with_dupes = 0
    for path in raw:
        df = pd.read_csv(path, parse_dates=["Datetime"])
        d = int(df["Datetime"].duplicated().sum())
        total_dupes += d
        files_with_dupes += 1 if d else 0

    # 10 of 12 files carry 4 duplicates each (two fall-back events x 2 rows).
    assert files_with_dupes == 10, f"expected 10 files with dupes, got {files_with_dupes}"
    assert total_dupes == 40, f"expected 40 duplicate timestamps, got {total_dupes}"


def test_clean_py_does_not_dedup(project_root):
    """Guard the gap: scripts/clean.py performs NO deduplication.

    If someone adds dedup to clean.py this test fails, forcing them to look
    here - because dedup in clean.py would be TOO LATE to fix lags, which are
    computed upstream in book1. Order is the whole point.
    """
    src = (project_root / "scripts" / "clean.py").read_text()
    assert "drop_duplicates" not in src
    assert "duplicated" not in src
