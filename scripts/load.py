from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from sqlalchemy import DateTime, Float, Integer, String

import numpy as np

from database.db_connection import get_engine
from database.models import create_tables
from scripts.clean import CLEAN_PATH

# Table that the SmartGridLoad model defines (see database/models.py).
TARGET_TABLE = "smart_grid_load"

# NOTE: this is the ONLY loader for the database. scripts/ingest.py used to
# push the same CSV into a second table ("smart_grid_load_raw") that nothing
# read, while database/models.py declares "smart_grid_load". That module has
# been reduced to a thin alias so there is exactly one canonical table.

CHUNKSIZE = 50_000  # rows per chunk - keeps RAM flat on large cleaned files

# Which columns are Integers vs Floats so SQLAlchemy never guesses VARCHAR.
# NOTE: these MUST match the SQLAlchemy model column types in database/models.py.
_INTEGER_COLS = [
    "hour", "day_of_week", "month", "quarter", "year", "is_weekend",
    "is_holiday", "is_month_start", "is_month_end",
]
# Everything else numeric (incl. MW and all the lag/roll/fourier floats) is Float.
_FLOAT_COLS = [  # for readability only - defaults handled below.
    "MW",
    "hour_sin", "hour_cos", "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "load_lag_1h", "load_lag_24h", "load_lag_168h", "load_lag_720h",
    "load_roll_mean_24h", "load_roll_std_24h",
    "load_roll_mean_168h", "load_roll_std_168h",
    "fourier_sin_1", "fourier_cos_1", "fourier_sin_2", "fourier_cos_2",
]


def _sql_dtypes():
    """Build the dtype map passed to to_sql so INSERTs use real SQL types.

    Without this, SQLAlchemy + method='multi' guesses column types per chunk and
    can misjudge Datetime as VARCHAR - which then fails against a real
    'timestamp' column in Postgres with a long error.
    """
    dtypes = {"Datetime": DateTime, "region": String(20)}
    for c in _INTEGER_COLS:
        dtypes[c] = Integer
    for c in _FLOAT_COLS:
        dtypes[c] = Float
    return dtypes


def load_clean_data():
    engine = get_engine()

    # 1) Drop any old version of the table so we start fresh.
    with engine.begin() as conn:
        conn.exec_driver_sql(f"DROP TABLE IF EXISTS {TARGET_TABLE} CASCADE")

    # 2) Build the table structure from the model (create_tables in models.py).
    create_tables(engine)

    # 3) Stream the cleaned CSV into the DB, chunk by chunk.
    total = 0
    reader = pd.read_csv(CLEAN_PATH, chunksize=CHUNKSIZE)
    for i, chunk in enumerate(reader):
        chunk = chunk.dropna(subset=["MW"])  # safety: no target -> skip

        # Convert the Datetime strings into real Python datetime objects.
        # Postgres maps these cleanly onto the 'timestamp' column; passing raw
        # strings can cause a long psycopg type/binding error.
        if "Datetime" in chunk.columns:
            chunk["Datetime"] = pd.to_datetime(chunk["Datetime"]).dt.to_pydatetime()

        # Postgres rejects NaN/inf in numeric columns: turn them into NULL.
        # (The clean file currently has none, but this guards future files.)
        #
        # The previous guard was a no-op: s.fillna(np.nan) changes nothing, and
        # assigning np.nan back into a float column is what was already there -
        # INFINITY survived and Postgres rejected the row. Coerce to a real
        # float64 array first so inf becomes detectable, then write None so
        # SQLAlchemy emits NULL instead of "nan"::float.
        for col in chunk.columns:
            if col == "Datetime":
                continue
            vals = pd.to_numeric(chunk[col], errors="coerce").to_numpy(
                dtype="float64", na_value=np.nan
            )
            bad = ~np.isfinite(vals)
            if bad.any():
                chunk.loc[bad, col] = None

        chunk.to_sql(
            TARGET_TABLE,
            engine,
            if_exists="append",   # table already exists; just add rows
            index=False,
            method="multi",
            chunksize=1000,
            dtype=_sql_dtypes(),  # force correct SQL types (esp. Datetime)
        )
        total += len(chunk)
        print(f"[load] chunk {i + 1}: {len(chunk)} rows (total {total})")

    print(f"[load] Done. {total} cleaned records in '{TARGET_TABLE}' table")


if __name__ == "__main__":
    # Quick sanity look at the cleaned file before loading it.
    df_preview = pd.read_csv(CLEAN_PATH, nrows=5)
    print("columns:", df_preview.columns.to_list())
    load_clean_data()
