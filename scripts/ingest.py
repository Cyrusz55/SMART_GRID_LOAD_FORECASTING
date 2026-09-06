from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from database.db_connection import get_engine

# Read the CLEANED file (the pipeline is: clean.py -> ingest.py).
# Fall back to the raw merged file if the cleaned one does not exist yet.
RAW_CSV_PATH = PROJECT_ROOT / "clean_data" / "merged_long_df_clean.csv"
if not RAW_CSV_PATH.exists():
    RAW_CSV_PATH = PROJECT_ROOT / "data_raw" / "merged_long_df.csv"
TARGET_TABLE = "smart_grid_load_raw"

CHUNKSIZE = 50_000  # rows per chunk - keeps RAM flat


def ingest_raw_data():
    engine = get_engine()
    first_df = None

    print(f"[Ingest] Reading from: {RAW_CSV_PATH}")

    # Stream the file chunk by chunk and load each piece into the DB.
    # Only one chunk (50k rows) is in memory at any moment.
    reader = pd.read_csv(RAW_CSV_PATH, chunksize=CHUNKSIZE)
    for i, chunk in enumerate(reader):
        # Drop rows with no target (MW) - keeps the DB clean.
        chunk = chunk.dropna(subset=["MW"])

        # First chunk: replace any old table with the new one.
        # Later chunks: append to the just-created table.
        if i == 0:
            chunk.to_sql(
                TARGET_TABLE,
                con=engine,
                if_exists="replace",   # create fresh table on the first chunk
                index=False,
                chunksize=1000,
                method="multi",
            )
        else:
            chunk.to_sql(
                TARGET_TABLE,
                con=engine,
                if_exists="append",    # keep adding rows
                index=False,
                chunksize=1000,
                method="multi",
            )
        print(f"[Ingest] loaded chunk {i + 1}: {len(chunk)} rows")

    print(f"[Ingest] Done. Raw data loaded into '{TARGET_TABLE}' table")


if __name__ == "__main__":
    ingest_raw_data()
