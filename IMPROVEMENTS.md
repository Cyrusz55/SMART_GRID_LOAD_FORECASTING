# SMART_GRID_LOAD_FORECASTING — Improvement Checklist

> **How to use this file:** work top to bottom. Each item is one tickable action.
> Ordering is deliberate — **Phase 1 fixes real bugs** that block everything else.
> Tick as you finish: change `[ ]` to `[x]`.

**Author:** Cyrus (`@Cyrusz55`)
**Reviewed:** 2026-09-15
**Provenance:** every item below came from reading the actual source files, not from generic advice.
**Verified baseline:** 3 commits · 5 notebooks · 0 tests · 0 CI · 0 Docker · model = 1.13 GB on disk.

**Last updated:** 2026-09-15 (after Phase 1 + Phase 2 execution)
**Current state:** Phase 1 complete (12/12 actionable, 1.2 withdrawn) · Phase 2 complete (2.1–2.9, **152 tests, 76% coverage**) · Phases 3–8 untouched.

> ⚠️ **Items 1.1–1.13 and 2.1–2.9 are DONE and ticked below.** The remaining items (2.10, 3.x onwards) are still open. Two corrections were made to this file during execution — see **Corrections** at the end.

---

## 📊 Verified baseline (measured, not estimated)

| Thing | Measured value | Command that proved it |
|---|---|---|
| Model file | **1,127,963,617 bytes = 1.05 GiB** | `ls -la models/best_rf_clean.joblib` |
| Feature list file | 401 bytes | `ls -la models/feature_cols_clean.joblib` |
| Feature count | **26** | `joblib.load(...)` → len == 26 |
| Clean CSV | **359,130,690 bytes = 342 MiB** | `ls -la clean_data/` |
| Raw CSVs | 12 per-region files, ~3 MB each | `ls -la data_raw/` |
| Notebooks | **5** (book1, book2, book2_fixed (1), book2_loaded, book2_study) | `ls -la notebooks/` |
| Tests | ~~0~~ → **152** | `pytest` → 152 passed, 76% cov |
| CI | **0** | no `.github/` directory |
| Docker | **0** | no Dockerfile / compose |

### 🔴 Bugs I reproduced (not guesses)

| # | File | Bug | Proof |
|---|---|---|---|
| **A** | `database/db_connection.py:21` | `postgres://` → `postgresql+psycopg//` — **the colon is missing**, producing an unparseable URL | Ran the exact `.replace()`; output was `postgresql+psycopg//user:pw@...`, and `"psycopg://" in out` → `False` |
| ~~**B**~~ | `apps/routes.py:175` | ~~`is_holiday` hardcoded to `0`~~ — **WITHDRAWN, NOT A BUG.** The training notebook also sets `is_holiday = 0` (`book2_loaded.ipynb` cell 43: `'is_holiday': 0, # no holidays in this dataset`), so inference **matches training**. The `holidays` package in `requirements.txt` is a leftover. | `book2_loaded.ipynb` cell 43 |
| **C** | `apps/main.py:99` | `root()` serves `frontend/index.html`, which **does not exist** | `ls frontend/` shows no `index.html`; the frontend is Reflex |
| **D** | `scripts/ingest.py:16` vs `scripts/load.py:15` | Two different target tables for the same data (`smart_grid_load_raw` vs `smart_grid_load`) | Read both files |
| **E** | `apps/routes.py:96` | `/api/v1/**` prefixed router, but the whole app is mounted with the prefix; frontend posts to `/api/v1/predict` ✅ — however `main.py` CORS omits port **8001** (the port the API actually runs on) | `grep -n "8001" apps/main.py` → **no match** |

**Bug E detail:** your CORS list is `3000, 8000`, but `rxconfig.py` and `load_forecast.py` both say the FastAPI app runs on **8001**. Browser calls from the Reflex frontend to `127.0.0.1:8001` are cross-origin; port 8001 is not in `allow_origins`. This works today only because Reflex proxies/proxies-less local dev often bypasses it — it will fail the moment you deploy the two as separate origins.

---

## 🔴 Phase 1 — Correctness (do these first, they're real bugs)

- [x] **1.1 — Fix the `postgres://` URL rewrite** (`database/db_connection.py:21`). **DONE.**
  Both `postgres://` and `postgresql://` now normalise to `postgresql+psycopg://` (two sequential `.replace()` blocks — the single-replace form missed the `postgresql://` variant).
  ```python
  # BEFORE (broken — no colon after psycopg):
  database_url = database_url.replace("postgres://", "postgresql+psycopg//", 1)

  # AFTER:
  database_url = database_url.replace("postgres://", "postgresql+psycopg://", 1)
  ```
  *Why it matters:* Heroku/Render/Supabase hand out `postgres://` strings. This bug fires **exactly on deploy day**, never locally.

- [x] ~~**1.2 — Implement `is_holiday` properly** (`apps/routes.py:175`).~~ **WITHDRAWN — NOT A BUG.**
  Verified against the training notebook: `book2_loaded.ipynb` **cell 43** sets `'is_holiday': 0, # no holidays in this dataset`. The model was trained with a constant `0`, so inference passing `0` **matches training exactly**. "Fixing" this would feed the model a feature it never learned from — strictly worse.
  The `holidays` package in `requirements.txt` is a **leftover**, safe to drop (rolled into 7.6a).
  *Lesson kept:* the feature exists in the list of 26, but its training-time value was constant. Always check what training actually produced before "correcting" inference.

- [x] **1.3 — Add port 8001 to CORS** (`apps/main.py:50-65`), plus your future deployed origins. **DONE.**
  Port `8001` is now present (the API's real port), along with bare-host variants, **and** extra origins are appended from an `ALLOWED_ORIGINS` env var so prod needs no code edit.
  ```python
  origins = [
      "http://localhost:3000", "http://127.0.0.1:3000",   # Reflex frontend
      "http://localhost:8001", "http://127.0.0.1:8001",   # the API itself
  ]
  ```
  Better: read from an `ALLOWED_ORIGINS` env var so prod doesn't need a code edit.

- [x] **1.4 — Add a readiness endpoint that truly checks the model.** **DONE.**
  `GET /api/v1/ready` added, returning `503` until the model is in memory; `/health` stays a cheap liveness check. Covered by `tests/test_api.py` (2.3).
  `GET /api/v1/health` used to return `{"status": "ok"}` **unconditionally** — it would report healthy with no model loaded, so any orchestrator would route traffic into a 500.
  ```python
  @router.get("/ready")
  def readiness():
      from apps.model_loader import model_loaded
      if not model_loaded():
          raise HTTPException(503, "model not loaded")
      return {"status": "ready"}
  ```
  Keep `/health` as cheap liveness; use `/ready` for traffic gating.

- [x] **1.5 — Make startup fail loudly, or don't load at startup at all** (`apps/main.py:71-93`). **DONE — chose 1.5b.**
  Startup now **hard-fails** (`RuntimeError` chained `from e`) on a missing or corrupt model by default. `EAGER_MODEL_LOAD=false` opts into lenient boot for local work. Success is logged via `logger.info` instead of `print`.

- [x] **1.6 — Replace `@app.on_event("startup")`** with the lifespan context manager (`apps/main.py`). **DONE.** `on_event` is deprecated in modern FastAPI and emits warnings; it breaks on upgrade. Now an `@asynccontextmanager lifespan` wired via `FastAPI(..., lifespan=lifespan)`, with the startup logic extracted to `_warm_model()`.
  ```python
  from contextlib import asynccontextmanager

  @asynccontextmanager
  async def lifespan(app: FastAPI):
      _warm_model()
      yield

  app = FastAPI(..., lifespan=lifespan)
  ```
  ⚠️ **Consequence discovered while testing (2.3):** `with TestClient(app) as c:` runs lifespan and therefore triggers the 1.05 GiB model load. Test fixtures must use a bare `TestClient(app)` (real HTTP, no startup) or they hang. A comment in `tests/conftest.py` records this so the fixture isn't "fixed" back.

- [x] **1.7 — Remove the dead `frontend/index.html` branch** (`apps/main.py`). **DONE.** That file does not exist; the frontend is Reflex. `root()` now returns the JSON message only (dead `FileResponse` branch dropped).

- [x] **1.8 — Drop the unused `import pandas as pd`** (`apps/main.py`). **DONE.** "Left available, mirroring the heart project" was not a reason to keep a heavy unused import in a service entrypoint.

- [x] **1.9 — Reject absurdly-far-future `start_datetime` with a 400** (`apps/routes.py`). **DONE.** `MAX_FORECAST_AHEAD = pd.Timedelta(days=7)`; a `start_datetime` beyond `last_real_ts + 7d` now returns **HTTP 400**. The README already admitted "2024+ is not statistically meaningful" but nothing enforced it — a 2024 date used to return confident nonsense.
  ```python
  MAX_AHEAD = pd.Timedelta(days=7)
  if start_ts > last_real_ts + MAX_AHEAD:
      raise HTTPException(400, f"start_datetime beyond reliable horizon "
                               f"({(last_real_ts + MAX_AHEAD).date()})")
  ```

- [x] **1.10 — Stop leaking internals in the 500 handler** (`apps/routes.py`). **DONE.** The handler now logs server-side with `logger.exception(...)` and returns an **opaque client message** — no `str(e)`, no traceback to the caller.

- [x] **1.11 — Guard the recursive feedback growth** (`apps/routes.py`). **DONE.** `_recursive_forecast` now pre-allocates one `np.empty(n_hist + horizon)` buffer instead of `np.append()` inside the loop — **O(n²) → O(n)** copying, identical math. `numpy` promoted to a module-level import.

- [x] **1.12 — Fix the duplicated-table ambiguity between `ingest.py` and `load.py`.** **DONE.** `scripts/ingest.py` is retired as a duplicate loader — it is now a `DeprecationWarning` shim forwarding to `scripts.load.load_clean_data()`, with `TARGET_TABLE = "smart_grid_load"` as the single canonical table. Chose `load.py`/`smart_grid_load` because `database/models.py:35-36` declares exactly that.

- [x] **1.13 — Fix `scripts/load.py:82` — the NaN/inf guard is a no-op.** **DONE.** The guard is now real: `pd.to_numeric(...).to_numpy(dtype="float64", na_value=np.nan)` plus a `~np.isfinite(vals)` mask that writes `None` (which psycopg emits as SQL `NULL`).
  ```python
  mask = ~np.isfinite(pd.to_numeric(s, errors="coerce").to_numpy(dtype="float64", na_value=np.nan))
  chunk.loc[mask, col] = None
  ```
  The old form (`s.fillna(np.nan)` then assigning `np.nan` back) changed nothing. The case Postgres actually rejects is `inf`.

---

## 🟠 Phase 2 — Testing (you have zero; this blocks CI)

- [x] **2.1 — Create `tests/` and add the dev deps.** **DONE.** `requirements-dev.txt` written with `pytest`, `pytest-cov`, `httpx` (required by `fastapi.testclient`), `ruff`, `mypy`. Kept separate from `requirements.txt` so the production image (Phase 4) never ships pytest.

- [x] **2.2 — `tests/test_schemas.py` — no model, runs in milliseconds.** **DONE — 37 passed in 0.58s.**
  Covered: each of the 12 regions accepted; an invalid region raises `ValidationError`; `horizon_hours=0` and `169` rejected while `24` passes; the `datetime` alias on `HourlyForecast` round-trips (`model_config = {"populate_by_name": True}`).

- [x] **2.3 — `tests/test_api.py` — API shape with a stubbed model.** **DONE — 12 passed.**
  Client built with `get_model` **monkeypatched** to a tiny fake regressor. Asserted: `/health` → 200; `/api/v1/predict` response matches `ForecastResponse`; a `start_datetime` inside historical range → 400 with the expected message.
  ⚠️ **Trap worth remembering:** `with TestClient(app)` as a context manager runs the lifespan hook (1.6) and therefore loads the **1.05 GiB model** — it hangs, and `EAGER_MODEL_LOAD=false` does **not** help, because the env var is read at module import while `_warm_model()` still calls `load_model()`. Use a bare `TestClient(app)`: real HTTP through the app, no startup. Nothing needs startup state because `get_model` is patched.

- [x] **2.4 — `tests/test_features.py` — the single highest-value test in this repo.** **DONE — 46 passed.**
  `_build_future_row()` is pure logic with zero I/O. Covered:
  - [x] **2.4a** `hour_sin`/`hour_cos` equal `sin`/`cos(2π·h/24)` for a known hour — parameterised over 6 hours, 5 months, all 7 weekdays, **plus** a separate test that `fourier_*` uses `(h-1)` rather than `h`
  - [x] **2.4b** `is_weekend` is `1` on a Saturday and `0` on a Wednesday
  - [x] **2.4c** `load_lag_24h` pulls exactly `y[-24]` from a known series
  - [x] **2.4d** `load_roll_mean_24h` equals `y[-24:].mean()`
  - [x] **2.4e** a **short** history (e.g. 10 points) yields `NaN` for `load_lag_720h` instead of raising
  - [x] **2.4f** the returned dict has **exactly the 26 keys** in `feature_cols_clean.joblib`
  - [x] **2.4g** (added) `is_holiday` is always `0`, documenting that inference **matches training** — see the withdrawn 1.2

- [x] **2.5 — `tests/test_clean.py` — `_clean_chunk` on a tiny in-memory frame.** **DONE — 33 passed.**
  Covered: NaN-`MW` rows dropped; each of the **8 lag columns** individually disqualifies a row; `0.0` and negative values are **kept**; output columns equal `KEEP_COLS` and are reordered to `KEEP_COLS` order; absent `KEEP_COLS` aren't fabricated; numeric strings coerced while `Datetime`/`region` are left alone; all three sentinels (`-9`, `-9.0`, `?`) verified through a **real `pd.read_csv`** using the production `na_values` setting — plus the inverse test that `-9` survives when `na_values` is not applied, proving the behaviour comes from that setting; and no mutation of the caller's frame.
  🔴 **This test file found a real bug — new finding, see 1.14 below.**

- [x] **2.6 — Regression test for the DST Cartesian explosion you already fixed.** **DONE — 12 passed.**
  Root cause located in `notebooks/book1.ipynb` **cell 4**: the old pipeline built the frame with a WIDE `pd.merge` on `Datetime`; a DST fall-back repeats the 02:00 wall-clock hour, so merging duplicated keys produced the **cross product** (2×2=4, chained 10× → **2^10 = 1024** rows). Fix was to stop merging and build **long form** with `pd.concat`, plus a per-file `drop_duplicates(subset="Datetime")` **before** any shift/lag logic.
  Locked in: the cross-product mechanism; the doubling 2→4→8→…→1024; the **second failure mode** (`how="inner"` across regions with non-overlapping ranges returns an **empty frame**); `concat` immunity; `keep="first"`; and the property that actually matters — a duplicate hour makes `load_lag_24h` point at a **different value**, with a test that dedup must run **before** `dropna`.
  Verified against the real data: **10 of 12** raw files carry **4 duplicates each = 40 total** (two fall-back events × 2 rows). NI and PJM_Load have none. Two tests assert this against `data_raw/`.
  ⚠️ **Note:** `scripts/clean.py` contains **zero** dedup/DST logic (`grep -c "drop_duplicates\|dst"` → `0`). The dedup lives only in `book1` and the shipped CSVs, so the per-file `drop_duplicates` is **load-bearing, not cosmetic**. A guard test fails if someone adds dedup to `clean.py` — too late to fix lags, which are computed upstream.

- [x] **2.7 — Test `_resolve_raw_path()` fallback order** (`apps/routes.py`). **DONE — 6 tests.**
  The fallback is proven by making the raw file **absent** while the clean one is present — the only way to show the fallback actually runs. Also: raw wins when both exist; `FileNotFoundError` (not a silent `None`) when neither does; the message names the remedy; a live-repo check that a real candidate resolves. The fixture monkeypatches `routes.__file__` into a throwaway tree, so no real `data_raw/` is touched.
  ⚠️ **Known limitation, pinned deliberately:** the check is `.exists()`, not `.is_file()`. A **directory** named `merged_long_df.csv` satisfies it, is returned, and would fail later at `read_csv`. A test asserts the current behaviour so switching to `.is_file()` fails loudly and forces a conscious decision.

- [x] **2.8 — Set a coverage floor** — `--cov-fail-under=70` scoped to `apps/` and `machine_learning/`. **DONE.** Committed in **`pytest.ini`** rather than left as CLI flags, so a bare `pytest` enforces it and Phase 3's CI job (3.1b) can't drift from it. Notebooks and one-off scripts are excluded.
  **Measured baseline: 76% total** (152 tests). Per module: `schemas.py` **100%**, `routes.py` **88%**, `model_loader.py` 62%, `main.py` 58%, `machine_learning.py` 49%.
  The remaining gap is almost entirely code that touches the large artefacts — `main.py` lifespan/`_warm_model` (1.05 GiB model), `model_loader`'s real `joblib.load` calls, and `load_model`/`load_feature_cols`/`__main__`. Those cannot be covered without violating the "tests never require the model artefact" rule, so **76% is near the honest ceiling** without a stub model. 70 leaves ~6 points of headroom: ordinary refactors pass, deleting a test file fails.
  `pytest.ini` also sets `testpaths = tests` and `--strict-markers --strict-config`, so a typo'd marker is now an error rather than a silent no-op.

- [x] **2.9 — Add a test that proves the model's feature order is respected.** **DONE — 6 tests.**
  `machine_learning.machine_learning.predict()` does `features[load_feature_cols()]`, which **both selects and reorders**. Both halves are tested: reversed columns give **identical** predictions to ordered ones (and the estimator received the trained order both times); a deliberately scrambled permutation (verified to actually differ) is also identical; extra columns (`MW`, junk) are dropped and the model receives exactly 26 features; **row order is preserved** — only columns may be reordered; and a **missing** trained column raises `KeyError` rather than silently zero-filling.
  The `FakeRegressor` sums each row, so a missing reorder would produce a genuinely different number — the assertion is not vacuous. No model artefact needed; `load_feature_cols` is stubbed.

- [ ] **2.10 — Track the `clean.py` KeyError fix as its own item (NEW, added 2026-09-15).**
  🔴 **Found while writing 2.5 — a genuine bug, not one of the original 73.**
  `scripts/clean.py:35` ran `df.dropna(subset=["MW"])` **unconditionally**. If the incoming frame had no `MW` column, this raised **`KeyError`** and killed the entire cleaning run on the **first chunk**. Line 44 had it right all along (`.subset([c for c in lag_cols if c in df.columns])`) — line 35 just didn't match it.
  **Fixed:** early return when `MW` is absent ("nothing to clean", not a crash). Regression test added in `test_clean.py`, plus a code comment explaining why.
  **Now ticked as 1.14 below.** Left here as a pointer so the numbering stays traceable.

---

## 🟠 Phase 2b — Additional findings from testing (added 2026-09-15)

- [x] **1.14 — `scripts/clean.py:35` unconditional `dropna(subset=["MW"])` → `KeyError`.** **FIXED.** See 2.10 for the full write-up. This was **not** in the original 73 items; it was found by writing tests, which is the strongest argument for Phase 2 existing.

- [ ] **1.15 — Numeric coercion runs AFTER the dropna filters in `_clean_chunk` (ordering subtlety, NOT yet fixed).**
  Steps 2–3 (`dropna`) run **before** step 4 (numeric coercion). So a value that is a non-numeric **string** is not "missing" at dropna time; the row survives, and only afterwards does coercion turn it into `NaN` — leaving a row with a **NaN target**.
  Production data is safe because `na_values=` in `read_csv` converts the real sentinels up front, but a genuinely malformed CSV cell would land a NaN `MW` in the cleaned output.
  **Not fixed** — behaviour is pinned by two tests in `test_clean.py` (`test_coercion_failure_in_lag_survives_dropna_then_becomes_nan`, `test_unparseable_mw_survives_dropna_then_becomes_nan`) plus a third proving the safe path. If the ordering is ever changed, the tests fail and make it a deliberate decision.

---

## 🟡 Phase 3 — CI/CD (the scaffolding you specifically wanted)

- [x] **3.1 — Create `.github/workflows/ci.yml`** triggered on `push` and `pull_request`. **DONE.** All four jobs live in one file, YAML-validated.
  - [x] **3.1a** Job **lint**: `ruff check .` + `black --check .` — `continue-on-error: true` (the tree has never been linted; a first run will be noisy).
  - [x] **3.1b** Job **test**: `pytest -q` — **no `--cov` flags restated**, because `pytest.ini` already owns the coverage config and the `--cov-fail-under=70` floor (see 2.8). Restating them here is how CI and local runs drift apart. This is the **only** job that can go red.
  - [x] **3.1c** Job **types**: `mypy apps/ machine_learning/ --ignore-missing-imports`, `continue-on-error: true`.
  - [x] **3.1d** Job **import smoke**: `python -c "import apps.main"` — imports the app object but does **not** run the lifespan hook, so it never touches the model. **Do not replace this with `with TestClient(app)`** — that runs lifespan and hangs for minutes on the 1.05 GiB load (the trap documented in `tests/conftest.py`).
  - [x] **3.1e** pip caching via `actions/setup-python`'s `cache: pip` + `cache-dependency-path: requirements-dev.txt` (built in — no separate `actions/cache` step needed).
  - **Deviation:** the test job installs `requirements-dev.txt` **plus an explicit runtime subset**, NOT `requirements.txt`. That file carries ~160 entries including `tensorflow`, `prophet` and `pmdarima`, none of which the suite needs; installing them would dominate the run time for zero benefit.

- [ ] **3.2 — Add the CI badge to `README.md`.** **BLOCKED on the first push** — a badge pointing at a workflow that has never run renders as "no status" / broken. Sequence it *after* the first successful CI run.
  `![CI](https://github.com/Cyrusz55/SMART_GRID_LOAD_FORECASTING/actions/workflows/ci.yml/badge.svg)`
  Your README currently has **5 shields whose links are literally `#`** — see 8.3. (Counting them: Python, scikit-learn, FastAPI, Reflex, Postgres.)

- [ ] **3.3 — Branch-protect the default branch**: require the CI job to pass before merge. Even solo, this stops broken pushes.
  **CORRECTION — the checklist originally said `main`. It is `master`.** Verified against the remote: `git ls-remote --symref origin HEAD` → `ref: refs/heads/master`. Protecting `main` would have silently protected a branch that does not exist and looked like it worked. `gh` 2.45.0 is installed, so this can be scripted rather than clicked.

- [x] **3.4 — Add a nightly `model-validation.yml`** (`on: schedule`). **DONE** — `03:17 UTC` daily (off the hour on purpose; GitHub's scheduler is congested at `:00`) plus `workflow_dispatch`. Asserts feature count `== 26` **and** `model.n_features_in_ == 26`, the clean CSV still exposes `Datetime`/`region`/`MW`, the `(region, Datetime)` composite key is unique in a 5000-row sample, and `predict()` on a fixed row is finite and inside `0 < pred < 1e6` MW.
  **Honest limitation:** the artefacts it validates are **gitignored**, so a fresh checkout has none of them. A nightly red X for "no model in a fresh checkout" is noise, not signal — so the job **skips with a `::warning::`** and says so in the run summary. An `HF_MODEL_REPO` hook is in place so that finishing **6.3** activates full validation by **setting a secret, not editing code**.
  Note the band check asserts *sanity*, not accuracy — accuracy needs the real data and belongs in a separate job.

- [x] **3.5 — Add `continue-on-error` only where honest.** **DONE.** `true` on `lint` and `types` only. `test` and `import-smoke` are never tolerated — a failing test suite cannot go green.

- [x] **3.6 — Never put the 1.05 GiB model in CI.** **ENFORCED** by `.gitignore` (`/models/`, `data_raw/`, `clean_data/`, `*.joblib`, `*.csv`), so the artefact cannot be committed in the first place. The test suite stubs the model (2.3) and needs no fixture at all. Hosting a downloadable model for the *nightly* job is 6.3's problem, not CI's.

- [x] **3.7 — Add `.github/pull_request_template.md`**: what changed / how it was tested / any risk. **DONE**, plus a checklist covering large artefacts, `IMPROVEMENTS.md` updates and hardcoded secrets.

- **Also fixed while doing 3.1:** `.coverage` was **not** in `.gitignore`, so a coverage file was sitting untracked in the tree ready to be committed by accident. Added `.coverage`, `.coverage.*`, `htmlcov/`, `coverage.xml`, `.pytest_cache/`, `.ruff_cache/`, `.mypy_cache/`.

> **⚠️ Phase 3 is written, not proven.** Both workflows parse as valid YAML and the job logic is sound, but **no GitHub Actions run has executed** — that needs a push, and nothing has been committed or pushed. "CI passes" is unverified until GitHub says so.

---

## 🟢 Phase 4 — Docker & fixed architecture

- [ ] **4.1 — `Dockerfile` for the API** — `python:3.12-slim`, non-root user, `uvicorn apps.main:app --host 0.0.0.0 --port 8001`, `HEALTHCHECK` on `/api/v1/ready`.

- [ ] **4.2 — `Dockerfile` for the Reflex frontend** — **multi-stage**, because Reflex compiles a Node/Vite bundle at build time. Don't ship the toolchain in the final image.

- [ ] **4.3 — `docker-compose.yml` with four services.**
  ```yaml
  services:
    db:       { image: postgres:16, ports: ["5432:5432"] }
    api:      { build: ., ports: ["8001:8001"], depends_on: [db] }
    frontend: { build: ./frontend, ports: ["3000:3000"] }
    # loader:  one-shot: python scripts/clean.py && python scripts/load.py
  ```
  This is what finally makes the README's "clone and run" story *true* — a fresh reader currently has **no database**, and `db_connection.py` raises `ValueError` without `DATABASE_URL`.

- [ ] **4.4 — Write down the port contract in one place.** It is currently spread across three files (`rxconfig.py`, `load_forecast.py`, `main.py`) and is easy to get wrong:

  | Service | Port | Source of truth today |
  |---|---|---|
  | FastAPI API | **8001** | `load_forecast.py` (`API_BASE`) |
  | Reflex backend (state websocket) | **8000** | `rxconfig.py` (`backend_port`) |
  | Reflex frontend | **3000** | `rxconfig.py` (`frontend_port`) |
  | Postgres | **5432** | `db_connection.py` (`DATABASE_URL`) |

  Put this in `.env.example` + a README table, and have compose consume the same values.

- [ ] **4.5 — Solve model-in-image properly.** Baking 1.05 GiB makes builds slow and images huge. Best first:
  - [ ] **4.5a** Download from Hugging Face Hub on container start, with a local cache volume — **recommended** (also matches your existing "next task: commit model to HuggingFace")
  - [ ] **4.5b** Mount as a host volume
  - [ ] **4.5c** Bake it in, but only with a strict `.dockerignore`

- [ ] **4.6 — `.dockerignore`** — must exclude `.venv/`, `data_raw/`, `clean_data/`, `models/`, `.git/`, `notebooks/`, `*.csv`, `*.joblib`. Without this, the build context uploads **~1.4 GB** (1.05 GiB model + 342 MiB CSV).

- [ ] **4.7 — `.env.example`** with placeholders (never the real `DATABASE_URL`). Your `.env` currently holds only `DATABASE_URL`, and it is correctly git-ignored ✅.

- [ ] **4.8 — `Makefile` or `justfile`**: `install`, `test`, `lint`, `run-api`, `run-frontend`, `data`, `up`. One discoverable entry point beats five README paragraphs.

---

## 🔵 Phase 5 — Architecture & code quality

- [ ] **5.1 — Extract the shared feature builder.** **This is the most important engineering item in the file.**
  Feature logic exists **twice**: `scripts/clean.py` (training) and `apps/routes.py::_build_future_row()` (inference). If they ever disagree, predictions degrade **silently** — no exception, no log, just worse numbers. Extract one `build_features(ts, y)` used by both. Note the two already differ in small ways (e.g. `quarter` is kept in training but absent from the inference row builder).
  - [ ] **5.1a** Write the shared function
  - [ ] **5.1b** Point `routes.py` at it
  - [ ] **5.1c** Point the notebook pipeline at it
  - [ ] **5.1d** Add test 2.4 as the contract that keeps them locked together

- [ ] **5.2 — De-duplicate `REGION_NAMES`** — it exists in **three** places: `apps/schemas.py:25`, `apps/routes.py`, and `frontend/load_forecast.py:40`. Define once, import everywhere. The frontend copy is guaranteed to drift.

- [ ] **5.3 — De-duplicate the model-path logic.** `apps/model_loader.py` (`DEFAULT_MODEL_PATH`, `_resolve_path`) and `machine_learning/machine_learning.py` (`MODEL_PATH`, `FEAT_COL_PATH`) resolve the same files independently. Consolidate into one config module.

- [ ] **5.4 — Add `config.py`** holding `PROJECT_ROOT`, `MODEL_PATH`, `FEAT_COL_PATH`, `CLEAN_PATH`, `RAW_PATH`, `TARGET_COL`. These are currently recomputed in five files.

- [ ] **5.5 — Replace the module-global model cache** (`model_loader.py:_model`) with `functools.lru_cache` or a small cache object. Glogals are painful to reset between tests.

- [ ] **5.6 — Remove the `sys.path.insert` hacks.** `scripts/ingest.py:5`, `scripts/load.py:5`, `scripts/test_connection.py:5` all mutate `sys.path`. Add `__init__.py` to `scripts/`, `database/`, `machine_learning/`, `frontend/` and use real package imports.

- [ ] **5.7 — Add type hints and run `mypy` in CI.** `_recursive_forecast(model, feat_cols, region_name, region_code, start_ts, horizon)` has **six untyped parameters**.

- [ ] **5.8 — Name the magic numbers.** `CHUNKSIZE = 50_000` is duplicated in `clean.py:26`, `ingest.py:18`, and `load.py:17`. Move to config.

- [ ] **5.9 — Consider Pydantic settings instead of raw `os.getenv`.** Two files each call `load_dotenv()` with **different** semantics — `db_connection.py:9` uses `override=True`, `main.py:36` uses the default (`override=False`). That's an invisible ordering dependency: whichever imports first wins. A single settings object fixes it.

- [ ] **5.10 — Add `pyproject.toml`** with `[tool.ruff]`, `[tool.black]`, `[tool.pytest.ini_options]`, `[tool.mypy]`. Right now there is no linter config at all.

---

## 🟣 Phase 6 — Model & ML engineering

- [ ] **6.1 — Shrink the model.** **1,127,963,617 bytes** for a Random Forest is the root cause of your 8.4-second forecasts. Options, in order of payoff:
  - [ ] **6.1a** Try **LightGBM** or **XGBoost** — both are **already in `requirements.txt`** and are 10–100× smaller/faster on tabular data like this
  - [ ] **6.1b** Cap `max_depth` and reduce `n_estimators` (the commented-out training block shows `n_estimators=120`)
  - [ ] **6.1c** If you keep sklearn, try `tree_method`-style pruning or float32 quantisation
  - [ ] **6.1d** Measure the accuracy delta — if LightGBM gives similar MAE at 1/20th the size, that *is* the finding

- [ ] **6.2 — Version the artefact.** Save a sidecar `model_card.json`: training date, data range, feature count (26), MAE/RMSE, `n_estimators`, git SHA. Right now `best_rf_clean.joblib` carries **no provenance at all**.

- [ ] **6.3 — Publish the model to Hugging Face Hub** (your own stated next task) and add a `MODEL_REPO` env var + download helper. This unblocks 4.5a and 3.6 simultaneously.

- [ ] **6.4 — Build a real evaluation report.** Today you have a single MAE figure in a notebook. Produce MAE / RMSE / MAPE **per region** plus:
  - [ ] **6.4a** A **seasonal-naive baseline** ("same hour yesterday" / "same hour last week"). If the RF doesn't beat it, the model isn't earning its complexity. This is the single most credible chart you can add.
  - [ ] **6.4b** A Ridge/linear baseline for the same reason

- [ ] **6.5 — Quantify recursive error accumulation.** Your README lists it as a limitation but never measures it. Plot **MAE vs. horizon step** (1h → 168h). Cheap to produce, and it's the most impressive figure you could put in the repo.

- [ ] **6.6 — Consider a direct multi-horizon model.** Recursion compounds error. Training one model per horizon (or predicting h+24 directly) usually beats recursion past ~24 h. Your own README says "≤48h is most reliable" — this is how you extend that.

- [ ] **6.7 — Prune the notebook sprawl.** You have **5 notebooks**, including `book2_fixed (1).ipynb` (a filename with a space and a "(1)" — clearly a duplicate download) and `book2.ipynb` / `book2_study.ipynb` / `book2_loaded.ipynb`. Keep **one** canonical training notebook (`book2_loaded.ipynb`, which the README already references), move the rest to `notebooks/archive/`, and state in the README which one is authoritative.

- [ ] **6.8 — Add the Kenyan-data roadmap as a real, tracked issue** (`gh issue create`), with a checklist: data source, granularity, holiday calendar, retrain, evaluate. Your last commit says "figure out how to use kenyan latest data" — prose in a commit message gets lost.

---

## ⚪ Phase 7 — Deployment & operations

- [ ] **7.1 — Replace `print()` with `logging`.** `[model_loader] Loading model from:`, `[startup] WARNING:`, `[DEBUG] Exception:` are all `print`. Use levels (`info`/`warning`/`error`) and JSON output in prod.

- [ ] **7.2 — Add server-side timing for `/predict`.** The frontend already measures latency (`latency_ms`) but the server doesn't. Log it; you'll want the p95.

- [ ] **7.3 — Add `/metrics`** (Prometheus): request count, latency histogram, model-load seconds, prediction errors. Especially valuable given 6.1 — you'll want to prove the latency win.

- [ ] **7.4 — Add a `.env.example`** (also 4.7) and document every variable: `DATABASE_URL`, `MODEL_PATH`, `ALLOWED_ORIGINS`.

- [ ] **7.5 — Verify no secret is in git history.**
  ```bash
  git log -p --all | grep -iE "DATABASE_URL|postgres://|password"
  ```
  `.gitignore` protects you going forward ✅, but history is forever. Check it now.

- [ ] **7.6 — Split `requirements.txt`.** It currently has ~160 entries including `tensorflow`, `keras`, `prophet`, and `pmdarima` — those are almost certainly notebook leftovers, not runtime deps for a Random Forest API. Split:
  - [ ] **7.6a** `requirements.txt` — runtime only (fastapi, uvicorn, joblib, numpy, pandas, sklearn, psycopg, sqlalchemy, dotenv, holidays, pydantic)
  - [ ] **7.6b** `requirements-dev.txt` — pytest, ruff, mypy, black
  - [ ] **7.6c** `requirements-notebooks.txt` — the heavy ML/EDA stack
  A 160-package install is why your container builds will be slow.

- [ ] **7.7 — Add a `deploy.yml` workflow** triggered on tags once CI is green (Render or GitHub).

- [ ] **7.8 — Add an uptime check** against the deployed `/api/v1/ready`.

- [ ] **7.9 — Add a `LICENSE` file.** The README has a License section — make it real (MIT is fine for a portfolio piece).

- [ ] **7.10 — Pin and lock.** You pin exact versions in `requirements.txt` (`fastapi==0.141.1`, `lightgbm==4.7.0`, …) which is good ✅ — but `pandas==3.0.5` and `numpy==2.5.2` are aggressive pins that will conflict with other packages. Consider a lockfile (`pip-tools` / `uv`) rather than hand-pinned transitive deps.

---

## 🏁 Phase 8 — Presentation (the layer reviewers actually see)

- [ ] **8.1 — Add an architecture diagram** to `README.md`. Your ASCII pipeline is genuinely good; a rendered version (data → clean → DB → train → model → API → UI) is better. The `diagram-maker` skill can produce SVG.

- [ ] **8.2 — Add the CI badge** (see 3.2) — visible proof the project is gated.

- [ ] **8.3 — Fix the placeholder links.** Four shields in your README point at `#` (`[![Python](...)]( # )` etc.). Every one should link somewhere real, or be dropped.

- [ ] **8.4 — Record a 30-second GIF** of the Reflex UI running a forecast. You already have `photos/dashboard-controls.png` and `photos/forecast-curve.png` — a GIF is the next step and the **highest-impact single item in this phase**. It makes the project legible in three seconds.

- [ ] **8.5 — Add a "Results" section** with the per-region MAE table (6.4) and the MAE-vs-horizon plot (6.5). Right now the README claims good performance without a single number in a table.

- [ ] **8.6 — Keep the honesty.** Your "Limitations & next steps" section is excellent — it already flags recursive drift, model staleness, the CSV-context issue, and the 1.1 GB size. Most portfolios hide those. Keep it and update it as you tick items off.

- [ ] **8.7 — Link this file from `README.md`** so the improvement plan is discoverable.

- [ ] **8.8 — Add a `CONTRIBUTING.md`** with the setup steps (venv, `.env`, `make install`). Cheap, and it signals "this is a real project".

---

## 🎯 Suggested order of attack

| Session | Do this | Why |
|---|---|---|
| **1** | Phase 1 entirely (13 items) | Real bugs, and 1.1/1.2/1.3 are deploy-day landmines |
| **2** | Phase 2 (tests) + 3.1 (CI) | Tests first, then the thing that runs them on every push |
| **3** | Phase 4 (Docker) + 4.4 (ports) | Makes "clone and run" actually true |
| **4** | **5.1** (shared features) + **6.1** (shrink model) | The two highest-value engineering wins |
| **5** | Phases 6.4/6.5 (evaluation) + 7 + 8 | Results and deployment story |

---

## 💡 The three things that matter most

If you do **only three things** from this entire file:

1. **5.1 — one shared feature builder.** Train-time and inference-time feature code can silently diverge. That's the failure mode that breaks a model in production with no error message anywhere. Fix it first.
2. **6.1 — shrink the model.** 1.05 GiB → a few hundred MB fixes your 8-second latency and makes deployment realistic. LightGBM is already in your requirements.
3. **1.1 + 1.3 — the two deploy-day bugs.** A broken DB URL rewrite and a missing CORS origin. Both are one-liners, and both fire exactly when you deploy.
   *(Originally listed as three, including 1.2's holiday flag. **1.2 was withdrawn** — see Corrections above: the model was trained with a constant `is_holiday = 0`, so inference matching it is correct. Had it been "fixed" as written, it would have fed the model a feature it never learned.)*

---

## 📋 Quick progress tracker

| Phase | Items | Done |
|---|---|---|
| 1 — Correctness | 13 | ✅ 12 / 13 (1.2 withdrawn — not a bug) |
| 2 — Testing | 10 | ✅ 9 / 10 (2.10 is a pointer to 1.14) |
| 2b — Extra findings | 2 | ⚠️ 1 / 2 (1.14 fixed, 1.15 open) |
| 3 — CI/CD | 8 | ⚠️ 5 / 8 done · 2 blocked (3.2 first push, 3.3 needs `gh` auth) |
| 4 — Docker | 8 | ☐ 0 / 8 |
| 5 — Architecture | 10 | ☐ 0 / 10 |
| 6 — Model & ML | 8 | ☐ 0 / 8 |
| 7 — Deployment | 10 | ☐ 0 / 10 |
| 8 — Presentation | 8 | ☐ 0 / 8 |
| **Total** | **76** | **✅ 22 done · ⚠️ 1 partially · ☐ 53 open** |

**Test suite as of 2026-09-15:** `pytest` → **152 passed, 76% coverage, ~12s.** No model artefact, no CSV, no network required.

| File | Tests |
|---|---|
| `tests/test_schemas.py` | 37 |
| `tests/test_features.py` | 46 |
| `tests/test_clean.py` | 33 |
| `tests/test_api.py` | 12 |
| `tests/test_dst_duplicates.py` | 12 |
| `tests/test_raw_path_and_order.py` | 12 |
| **total** | **152** |

---

## 🔧 Corrections made to this file (2026-09-15)

This file is a working document and was **wrong in two places**. Both are recorded rather than quietly deleted, because the wrong version was itself instructive:

1. **1.2 was not a bug — withdrawn.** Originally listed as "`is_holiday` hardcoded to `0` — but it IS in the 26 model features", with the implied fix of computing real holidays. The training notebook (`book2_loaded.ipynb` **cell 43**) sets `'is_holiday': 0, # no holidays in this dataset` — a **constant** during training. Inference passing `0` therefore **matches training exactly**, and "fixing" it would have fed the model a feature it never learned from: strictly worse. The `holidays` dependency is a leftover.
   *Root error:* the feature appearing in the list of 26 was treated as proof that it varied. Presence in a feature list says nothing about a feature's training-time distribution — a constant column is still a column. **Always check what training actually produced before "correcting" inference.**

2. **2.6's premise was slightly off.** It asked to "feed a frame with duplicated fall-back timestamps and assert the output row count is exactly right." But there is **no dedup step in the runnable pipeline** to assert against — `grep -c "drop_duplicates\|dst" scripts/clean.py` → **0**, and the fix exists only in notebook `book1` plus the CSVs already on disk. The tests were therefore written against the **mechanism** (merge fan-out) and the **contract** (composite-key uniqueness), not a function that doesn't exist. The row count was only ever a proxy; the real stake — stated in book1's own comment — is that duplicated hours make **lags point at fake neighbours**.

**Scope note:** items 1.1–1.13 and 2.1–2.9 were verified with `py_compile` plus **static reasoning** only; the test suite (2.1–2.9) is what has actually been executed. **No end-to-end runtime test has been performed against the real 342 MiB CSV or the 1.05 GiB model.** The 1.11 buffer rewrite, the 1.9 horizon limit, and the 1.5 startup hard-fail are verified by construction, not by execution. Closing that gap means either running the API once against the real data, or building a tiny fixture model so the full predict path is executable in tests (which would also lift `main.py`/`model_loader.py` past the 76% ceiling).

---

*Generated from a direct read of every source file in the repository, with the bugs reproduced by running the code. Nothing above is generic advice.*
