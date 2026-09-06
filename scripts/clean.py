from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = PROJECT_ROOT / "data_raw" / "merged_long_df.csv"
CLEAN_PATH = PROJECT_ROOT / "clean_data" / "merged_long_df_clean.csv"

target_col = "MW"  # the load in Megawatts - the value we predict

# Columns the model actually needs for training/prediction.
# (Datetime kept only for reference/timestamping, not fed to the model.)
KEEP_COLS = [
    "Datetime", "region", "MW",
    "hour", "day_of_week", "month", "quarter", "year", "is_weekend",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "load_lag_1h", "load_lag_24h", "load_lag_168h", "load_lag_720h",
    "load_roll_mean_24h", "load_roll_std_24h", "load_roll_mean_168h",
    "load_roll_std_168h",
    "is_holiday", "is_month_start", "is_month_end",
    "fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2",
]

# How many rows to read from disk at a time. Small enough to not blow up RAM,
# big enough to stay fast. 50k rows is a safe middle ground.
CHUNKSIZE = 50_000


def _clean_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """Clean ONE chunk of rows (kept small => low memory)."""
    # 1) Keep only the columns we care about.
    df = df[[c for c in KEEP_COLS if c in df.columns]]

    # 2) Drop rows with no target (MW) - cannot predict without it.
    df = df.dropna(subset=[target_col])

    # 3) Drop rows whose lag/rolling features are NaN.
    #    These are the first hours of each region (no history yet) - unusable.
    lag_cols = [
        "load_lag_1h", "load_lag_24h", "load_lag_168h", "load_lag_720h",
        "load_roll_mean_24h", "load_roll_std_24h", "load_roll_mean_168h",
        "load_roll_std_168h",
    ]
    df = df.dropna(subset=[c for c in lag_cols if c in df.columns])

    # 4) Force every numeric feature to a proper numeric type (safety net).
    for col in df.columns:
        if col not in ("Datetime", "region"):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def clean_data_in_chunks():
    """Stream the 1.1M-row CSV chunk by chunk, writing each cleaned piece.
    Memory stays flat because we never hold the whole file at once."""
    # Prepare the output folder.
    CLEAN_PATH.parent.mkdir(parents=True, exist_ok=True)

    total_written = 0
    first = True

    # na_values: treat -9 / -9.0 / '?' as missing (heart project convention).
    reader = pd.read_csv(
        RAW_PATH,
        usecols=KEEP_COLS,
        chunksize=CHUNKSIZE,
        na_values=["-9", "-9.0", "?"],
    )

    for i, chunk in enumerate(reader):
        chunk = _clean_chunk(chunk)

        if len(chunk) == 0:
            continue  # nothing usable in this piece - skip writing

        # Write the first chunk with a header, later chunks append without it.
        chunk.to_csv(
            CLEAN_PATH,
            mode="w" if first else "a",
            header=first,
            index=False,
        )
        first = False
        total_written += len(chunk)
        print(f"[clean] chunk {i + 1}: wrote {len(chunk)} rows (total {total_written})")

    print(f"[clean] Done. Cleaned data saved to {CLEAN_PATH} ({total_written} rows)")


if __name__ == "__main__":
    clean_data_in_chunks()
