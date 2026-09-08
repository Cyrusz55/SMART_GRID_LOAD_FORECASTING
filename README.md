<div align="center">

# ⚡ Smart Grid Load Forecasting

**Hourly electrical load (MW) forecasting for PJM regions using a Random Forest model — a full end-to-end machine learning project with a FastAPI backend and a Reflex web frontend.**

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](#)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Random%20Forest-orange.svg)](#)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-green.svg)](#)
[![Reflex](https://img.shields.io/badge/Reflex-0.9.10-purple.svg)](#)
[![Postgres/Supabase](https://img.shields.io/badge/PostgreSQL-Supabase-336791.svg)](#)

</div>

---

## 📌 Overview

This project predicts the **hourly electrical load (in megawatts, MW)** for any of **12 PJM regions** (e.g. `AEP`, `COMED`, `DAYTON`, `PJM_Load`) from a small number of inputs: the **region**, a **start date/time**, and a **forecast horizon** in hours.

It is a complete, production-shaped ML pipeline that mirrors the classic *data → clean → store → train → serve* architecture:

- **Raw hourly load data** for multiple regions is merged, cleaned, and enriched with **calendar + cyclical + lag/rolling time features**.
- A **Random Forest regressor** is trained on ~1M rows with proper **chronological (time-ordered) splits** to avoid data leakage.
- The trained model is served through a **FastAPI** backend exposing a **`/api/v1/predict`** endpoint.
- A **Reflex (pure-Python)** web frontend lets the user pick a region + start time + horizon, then plots the **forecasted load curve**.

> 📚 Built as a hands-on learning exercise by mirroring the structure of a completed *heart-disease ML project* — every "why" was deliberately commented so the pipeline is easy to study and extend.

---

## 🖼️ Screenshots

<p align="center">
  <img src="photos/dashboard-controls.png" alt="Smart grid load forecasting dashboard — controls and result" width="420"/>
  <img src="photos/forecast-curve.png" alt="Forecasted electrical load curve" width="420"/>
</p>

---

## ✨ Features

- **12 PJM regions** supported with a single multi-region model.
- **Recursive multi-step forecasting** (1–168 hours): each hour's forecast feeds back as input for the next.
- **Rich feature engineering**: cyclical time encoding (hour/day/month), Fourier terms, lags (1h/24h/168h/720h), and rolling mean/std (24h/168h).
- **Chronological train/test split + walk-forward CV** to keep the evaluation honest for time-series data.
- **Chunked data processing** so the full dataset never needs to fit in memory at once (low-RAM friendly).
- **Cached model loading** — the ~1 GB model file is read from disk only once.
- **Modern web UI** built entirely in Python (Reflex + Plotly) with live forecast curves.
- **Interactive API docs** at `/docs` (Swagger UI).

---

## 🗂️ Project Structure

```
SMART_GRID_LOAD_FORECASTING/
├── data_raw/                  # Raw per-region hourly CSVs (from Kaggle PJM data)
├── clean_data/                # Merged + cleaned long dataframe (single CSV)
├── notebooks/                 # Jupyter study notebooks (EDA → training)
│   └── book2_loaded.ipynb     # Loaded-model study pipeline (Cell 1–21)
├── scripts/
│   ├── clean.py               # Clean + feature-engineer raw data (chunked)
│   ├── ingest.py / load.py    # Load cleaned rows into the database (Supabase)
│   └── test_connection.py     # DB connection sanity check
├── database/
│   ├── db_connection.py       # Postgres engine (psycopg), reads .env DATABASE_URL
│   └── models.py              # SQLAlchemy table model (smart_grid_load)
├── machine_learning/
│   └── machine_learning.py    # Load model + feature list; predict helpers
├── apps/                      # FastAPI backend
│   ├── main.py                # App entry, CORS, mounts /api/v1
│   ├── routes.py              # /health, /predict, /model-info
│   ├── schemas.py             # Pydantic request/response models
│   └── model_loader.py        # Cached joblib model loader
├── frontend/                  # Reflex pure-Python web UI
│   ├── load_forecast.py       # Page + state (controls → plotly chart)
│   └── rxconfig.py            # Reflex config (ports, CORS origins)
├── models/                    # Trained artifacts (git-ignored)
│   ├── best_rf_clean.joblib   # ~1.1 GB RandomForestRegressor
│   └── feature_cols_clean.joblib
├── photos/                    # Screenshots used in this README
└── requirements.txt
```

---

## 🧱 Pipeline (how it works)

```
 data_raw/*.csv ──► scripts/clean.py ──► clean_data/merged_long_df_clean.csv
                                              │
                                      database (Supabase / PostgreSQL)
                                              │
                                            models/ ──► machine_learning/
                        (trained Random Forest in  best_rf_clean.joblib)
                                              │
                                        apps/  (FastAPI /api/v1/predict)
                                              │
                                    frontend/  (Reflex UI + Plotly)
```

1. **Clean & feature-engineer** (`scripts/clean.py`) — streams each raw CSV in 50k-row chunks, keeps the target + feature columns, drops rows with unusable lag context, and engineers time-based features (`hour_sin/cos`, `month_sin/cos`, `load_lag_*`, `load_roll_*`, Fourier terms, …). Chunking keeps RAM flat on ~1M rows.
2. **Persist to the database** (`scripts/load.py`) — the cleaned dataframe is streamed into the `smart_grid_load` table on Supabase via SQLAlchemy (explicit `dtype` mapping avoids Pandas/psycopg timestamp pitfalls).
3. **Train** (notebooks) — a `RandomForestRegressor` is trained on a **chronological** split (no future leakage) and saved, along with its exact feature-column order, to `models/`.
4. **Serve** (`apps/`) — FastAPI loads the model **once** (cached) and exposes:
   - `GET  /api/v1/health` — liveness check
   - `POST /api/v1/predict` — returns hourly MW forecasts for a region + start time + horizon
   - `GET  /api/v1/model-info` — describes the loaded model
5. **Visualize** (`frontend/`) — a Reflex UI collects the inputs, calls `/predict`, and renders the forecast curve (Plotly) plus key summary cards (region, points, peak MW).

### 🧠 Feature engineering highlights

The model never sees raw timestamps — time is converted into numbers it can learn cyclical patterns from:

| Family | Features |
|--------|----------|
| Calendar | `hour`, `day_of_week`, `month`, `quarter`, `year`, `is_weekend`, `is_month_start/end` |
| Cyclical | `hour_sin/cos`, `day_of_week_sin/cos`, `month_sin/cos` |
| Fourier | `fourier_sin/cos_1/2` |
| Lags | `load_lag_1h`, `load_lag_24h`, `load_lag_168h`, `load_lag_720h` |
| Rolling | `load_roll_mean/std_24h`, `load_roll_mean/std_168h` |

Lag/rolling features give the model the **recent history** it needs — this is what makes short-horizon load forecasts accurate.

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- (Optional) PostgreSQL / Supabase for the DB-loading step
- Node.js ≥ 22.22 (only required to run the Reflex **frontend**)

### 1. Clone & install

```bash
git clone https://github.com/Cyrusz55/SMART_GRID_LOAD_FORECASTING.git
cd SMART_GRID_LOAD_FORECASTING

python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure the database (optional)

Create a `.env` in the project root with your Postgres connection string:

```env
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

> The connection rewrites the scheme to `postgresql+psycopg://` automatically (see `database/db_connection.py`).

### 3. Clean the raw data (optional, needs raw CSVs in `data_raw/`)

```bash
python scripts/clean.py          # data_raw/  →  clean_data/merged_long_df_clean.csv
python scripts/load.py           # loads the cleaned CSV into the smart_grid_load table
```

### 4. Serve the model — FastAPI backend

```bash
uvicorn apps.main:app --port 8001 --reload
```

- Interactive docs: http://127.0.0.1:8001/docs
- Health check: http://127.0.0.1:8001/api/v1/health

### 5. Run the Reflex frontend (separate terminal)

```bash
cd frontend
reflex run
```

Then open **http://localhost:3000**.

---

## 🔌 API Reference

### `POST /api/v1/predict`

**Request body:**

```json
{
  "region": "PJM_Load",
  "start_datetime": "2018-07-15T00:00:00",
  "horizon_hours": 24
}
```

**Response:**

```json
{
  "region": "PJM_Load",
  "forecasts": [
    { "datetime": "2018-07-15T01:00:00", "predicted_mw": 31042.56 },
    { "datetime": "2018-07-15T02:00:00", "predicted_mw": 29810.9 }
  ]
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `region` | string | — | One of the 12 PJM regions (see below) |
| `start_datetime` | datetime | — | First hour to forecast |
| `horizon_hours` | int | `24` | Hours to forecast ahead (`1`–`168`) |

**Supported regions (12):**
`AEP`, `COMED`, `DAYTON`, `DEOK`, `DOM`, `DUQ`, `EKPC`, `FE`, `NI`, `PJME`, `PJMW`, `PJM_Load`

> ⚠️ **Model validity window:** the trained model only knows patterns up to ~August 2018 (the end of its training data). Reliable forecasts extend only a short way beyond the last real data point — forecasting far into the future (e.g. 2024+) is **not statistically meaningful** and the API may reject `start_datetime` values that fall inside historical data.

---

## 🧪 Sample notebook

[`notebooks/book2_loaded.ipynb`](notebooks/book2_loaded.ipynb) walks the whole story — loading data, sanity checks, stationarity hints, chronological splits & walk-forward CV, loading the trained model, evaluation plots, residual analysis, feature importance, and finally **recursive future forecasting** (Cell 21, the same logic the API uses).

---

## 🛡️ Limitations & next steps

- **Recursive error accumulation** — each predicted hour is fed forward, so errors compound over long horizons. Short horizons (≤ 48h) are the most reliable.
- **Model staleness** — retrain periodically (e.g. a nightly scheduler) to stay current; the FastAPI `main.py` has a commented hook for this.
- **Historical context source** — production forecasting should read real recent history from the DB (`smart_grid_load`) rather than rebuilding context from the CSV each request.
- **Model size** — `best_rf_clean.joblib` is ~1.1 GB. For deployment, consider compressing/pruning it or hosting it on Hugging Face and downloading at startup (as the original heart-disease mirror project does).

---

## 👤 Author

**Cyrus** — Electrical & Electronics engineering student with a growing interest in data science and machine learning.

- 💻 GitHub: [@Cyrusz55](https://github.com/Cyrusz55)

---

## 📄 License

This project is for **educational purposes**. The PJM hourly load dataset (via Kaggle) is publicly available for research.
