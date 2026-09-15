"""
ingest.py
=========
DEPRECATED - use scripts/load.py instead.

History
-------
This module used to push the merged CSV into a table called
``smart_grid_load_raw``, while ``scripts/load.py`` pushed the very same file
into ``smart_grid_load`` - the table ``database/models.py`` actually declares.
Two loaders, two tables, one source file: the "raw" table was written by
nothing that ran in the documented pipeline and read by nothing at all, so it
silently accumulated duplicate data (or, more often, never existed).

Resolution (IMPROVEMENTS.md item 1.12)
--------------------------------------
``scripts/load.py`` is the single canonical loader, writing the canonical
table ``smart_grid_load``. This module is kept only so existing muscle memory
and any external references still resolve; it forwards to the real loader and
warns that it is deprecated.
"""
import sys
import warnings
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Re-exported for backwards compatibility with anything that imported these.
TARGET_TABLE = "smart_grid_load"

CSV_PATH = PROJECT_ROOT / "clean_data" / "merged_long_df_clean.csv"
if not CSV_PATH.exists():
    CSV_PATH = PROJECT_ROOT / "data_raw" / "merged_long_df.csv"

CHUNKSIZE = 50_000  # kept for compatibility; load.py owns the real value


def ingest_raw_data(*args, **kwargs):
    """Deprecated shim -> scripts.load.load_clean_data()."""
    warnings.warn(
        "scripts/ingest.py is deprecated and no longer loads anything itself; "
        "it forwards to scripts/load.py, which writes the canonical "
        "'smart_grid_load' table. Import scripts.load instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from scripts.load import load_clean_data

    return load_clean_data()


if __name__ == "__main__":
    print("[ingest] DEPRECATED: forwarding to scripts/load.py ...")
    ingest_raw_data()
