"""Reflex app — load_forecast (frontend).

Pure-Python frontend for the Smart Grid Load Forecasting FastAPI backend.

Run the BACKEND (unchanged) separately:
    uvicorn apps.main:app --reload --port 8000

Then run THIS frontend from the frontend/ folder:
    reflex run

Request payload matches apps/schemas.py ForecastRequest:
    {"region": str, "start_datetime": ISO datetime, "horizon_hours": int}

Expected response (ForecastResponse):
    {"region": str,
     "forecasts": [{"datetime": ISO, "predicted_mw": float}, ...]}
"""
from __future__ import annotations

from datetime import datetime
from typing import List, Optional

import requests
import reflex as rx
from plotly.graph_objects import Figure as PlotlyFigure
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Region list must match the backend REGION_NAMES / ForecastRequest Literal.
# ---------------------------------------------------------------------------
REGIONS = [
    "AEP", "COMED", "DAYTON", "DEOK", "DOM", "DUQ",
    "EKPC", "FE", "NI", "PJME", "PJMW", "PJM_Load",
]

# Backend URL. When you run `reflex run`, the frontend is served on 3000 and
# Reflex's OWN backend (state sync / websocket) runs on 8000.
# Your FastAPI app serves the /api/v1/predict endpoint on 8001 (run it yourself
# with `uvicorn apps.main:app --port 8001`). Change API_BASE if you host elsewhere.
API_BASE = "http://127.0.0.1:8001"
PREDICT_URL = f"{API_BASE}/api/v1/predict"


def _empty_figure() -> PlotlyFigure:
    """A blank plotly Figure used as the chart's default state."""
    # Style it to match the dark UI; empty data list => an empty chart until
    # go() replaces it with the real forecast figure.
    return go.Figure(
        data=[],
        layout={"template": "plotly_dark", "margin": {"t": 46, "b": 40, "l": 50, "r": 20}},
    )


class ForecastState(rx.State):
    """Holds whatever the user typed plus the forecast result."""

    # --- inputs ---
    region: str = "PJM_Load"
    start_date: str = "2018-07-15"     # YYYY-MM-DD
    start_time: str = "00:00"          # HH:MM
    horizon: int = 24

    # --- runtime flags / feedback ---
    loading: bool = False
    error: str = ""
    latency_ms: Optional[float] = None

    # --- result ---
    result_region: str = ""
    rows: List[List] = []              # each row: [datetime, predicted_mw]
    chart_figure: PlotlyFigure = _empty_figure()  # real plotly Figure

    # --- summary-card text (computed in go(), rendered as plain strings) ---
    summary_region: str = "—"
    summary_points: str = "—"
    summary_peak: str = "—"

    # --- explicit setters for the form controls ---
    # (This Reflex build does not auto-generate `set_<var>` handlers, so we
    # declare them by hand. Each takes the new value the widget emits.)
    def set_region(self, value: str):
        self.region = value

    def set_start_date(self, value: str):
        self.start_date = value

    def set_start_time(self, value: str):
        self.start_time = value

    def set_horizon(self, value):
        # Radix slider's on_change emits a LIST of floats (multi-thumb capable).
        # Single-thumb slider -> a one-element list; take its first value.
        if isinstance(value, (list, tuple)):
            value = value[0] if value else 24
        self.horizon = int(value)

    def go(self):
        """Call the backend /api/v1/predict and store the result."""
        # Reset previous state.
        self.loading = True
        self.error = ""
        self.latency_ms = None
        self.rows = []
        self.chart_figure = _empty_figure()
        self.summary_region = "—"
        self.summary_points = "—"
        self.summary_peak = "—"

        dt_str = f"{self.start_date} {self.start_time}:00"

        payload = {
            "region": self.region,
            "start_datetime": dt_str,
            "horizon_hours": self.horizon,
        }

        try:
            import time
            t0 = time.perf_counter()
            resp = requests.post(PREDICT_URL, json=payload, timeout=120)
            elapsed = (time.perf_counter() - t0) * 1000.0
            self.latency_ms = round(elapsed, 1)

            if resp.status_code != 200:
                raise RuntimeError(self._detail(resp))

            data = resp.json()
            self.result_region = data.get("region", self.region)
            forecasts = data.get("forecasts", [])

            # Each forecast: {"datetime": "...", "predicted_mw": 1234.56}
            self.rows = [
                [datetime.fromisoformat(f["datetime"]).strftime("%Y-%m-%d %H:%M"),
                 round(float(f["predicted_mw"]), 1)]
                for f in forecasts
            ]
            self.chart_figure = self._build_figure(forecasts)

            # Summary text for the cards (plain strings; we are in Python now).
            self.summary_region = self.result_region or "—"
            self.summary_points = str(len(self.rows))
            if self.rows:
                peak = max(self.rows, key=lambda r: r[1])
                self.summary_peak = f"{peak[1]:,.0f} MW @ {peak[0][-5:]}"
            else:
                self.summary_peak = "—"
        except Exception as exc:  # noqa: BLE001 - surface to user
            self.error = str(exc)
            self.rows = []
            self.chart_figure = _empty_figure()
            self.summary_region = "—"
            self.summary_points = "—"
            self.summary_peak = "—"
        finally:
            self.loading = False

    # --- helpers ---
    @staticmethod
    def _detail(resp) -> str:
        """Pull a readable message out of a non-2xx response."""
        try:
            data = resp.json()
            return str(data.get("detail", f"HTTP {resp.status_code}"))
        except Exception:  # noqa: BLE001
            return f"HTTP {resp.status_code}"

    @staticmethod
    def _build_figure(forecasts: List[dict]) -> PlotlyFigure:
        """Convert the raw forecast list into a real plotly Figure."""
        xs = [f["datetime"] for f in forecasts]
        ys = [float(f["predicted_mw"]) for f in forecasts]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="lines+markers",
                name="Forecast (MW)",
                line={"color": "#c9a84c", "width": 2.5},
                marker={"color": "#e8c87a", "size": 6},
            )
        )
        fig.update_layout(
            title="Forecasted electrical load",
            xaxis_title="Datetime",
            yaxis_title="MW",
            template="plotly_dark",
            margin={"t": 46, "b": 40, "l": 50, "r": 20},
        )
        return fig


# ---------------------------------------------------------------------------
# UI building blocks
# ---------------------------------------------------------------------------
def stat_card(label: str, value: str):
    return rx.vstack(
        rx.text(label, color="#a0937d", font_size="0.82rem", letter_spacing="0.08em",
                text_transform="uppercase", font_weight="700"),
        rx.text(value, font_size="1.6rem", font_weight="800", letter_spacing="-0.02em",
                color="#f5f0e8"),
        align="start",
        spacing="2",
        width="100%",
        padding="0.9rem 1rem",
        border="1px solid rgba(255,255,255,0.08)",
        border_radius="14px",
        background="rgba(255,255,255,0.03)",
    )


def page() -> rx.Component:
    state = ForecastState
    return rx.container(
        rx.vstack(
            # top bar
            rx.hstack(
                rx.hstack(
                    rx.box("RF", color="#1a1200", font_weight="800",
                           padding="0.55rem 0.8rem", border_radius="12px",
                           background="linear-gradient(135deg,#c9a84c,#e8c87a)"),
                    rx.vstack(
                        rx.text("Smart grid load forecasting", font_weight="800",
                                font_size="0.98rem"),
                        rx.text("Random Forest — trained on PJM hourly data",
                                color="#9b8f7c", font_size="0.85rem"),
                        spacing="1", align="start",
                    ),
                    spacing="3", align="center",
                ),
                rx.hstack(
                    rx.hstack(
                        rx.box(width="9px", height="9px", border_radius="999px",
                               background="#c9a84c"),
                        rx.text(f"Horizon: {state.horizon}h", font_size="0.88rem"),
                        spacing="2", border="1px solid rgba(255,255,255,0.12)",
                        padding="0.5rem 0.9rem", border_radius="999px",
                    ),
                    spacing="3",
                ),
                justify="between", align="center", width="100%", margin_bottom="0.5rem",
            ),

            # hero band
            rx.box(
                rx.vstack(
                    rx.text("REGIONAL LOAD ESTIMATE", color="#c9a84c", font_size="0.8rem",
                            font_weight="800", letter_spacing="0.16em",
                            text_transform="uppercase"),
                    rx.heading(
                        "Forecast hourly load in megawatts for any PJM region.",
                        font_size="2.1rem", line_height="1.05", letter_spacing="-0.03em",
                    ),
                    rx.text(
                        "Pick a region, a starting hour, and a horizon. The backend runs "
                        "a recursive forecast with the saved Random Forest model and "
                        "returns hourly MW predictions — shown here as a chart and a table.",
                        color="#b3a78f", max_width="52ch", line_height="1.6",
                    ),
                    align="start", spacing="3",
                ),
                width="100%", padding="1.6rem 1.6rem",
                border="1px solid rgba(255,255,255,0.12)",
                border_radius="20px",
                background="rgba(255,255,255,0.03)",
            ),

            # workspace: controls (left) + results (right)
            rx.grid(
                # ---- controls panel ----
                rx.vstack(
                    rx.text("Forecast controls", font_size="1.35rem", font_weight="800"),
                    rx.text("Fields below map to the backend ForecastRequest schema.",
                            color="#9b8f7c", font_size="0.92rem", line_height="1.5"),

                    # region
                    rx.vstack(
                        rx.text("Region", font_weight="600", font_size="0.95rem"),
                        rx.select(
                            REGIONS,
                            value=state.region,
                            on_change=ForecastState.set_region,
                            width="100%",
                        ),
                        align="start", spacing="2", width="100%",
                    ),

                    # date + time
                    rx.hstack(
                        rx.vstack(
                            rx.text("Start date", font_weight="600", font_size="0.95rem"),
                            rx.input(type_="date", value=state.start_date,
                                     on_change=ForecastState.set_start_date, width="100%"),
                            align="start", spacing="2", width="100%",
                        ),
                        rx.vstack(
                            rx.text("Start time (24h)", font_weight="600", font_size="0.95rem"),
                            rx.input(type_="time", value=state.start_time,
                                     on_change=ForecastState.set_start_time, width="100%"),
                            align="start", spacing="2", width="100%",
                        ),
                        spacing="4", width="100%",
                    ),

                    # horizon
                    rx.vstack(
                        rx.text("Horizon (hours)", font_weight="600", font_size="0.95rem"),
                        rx.slider(default_value=state.horizon, min=1, max=168, step=1,
                                  on_change=ForecastState.set_horizon, width="100%"),
                        align="start", spacing="2", width="100%",
                    ),

                    # submit
                    rx.button(
                        rx.cond(
                            state.loading,
                            rx.text("Forecasting…"),
                            rx.text("Run forecast"),
                        ),
                        on_click=ForecastState.go,
                        disabled=state.loading,
                        width="100%", min_height="2.8rem", color_scheme="gold",
                    ),

                    rx.cond(
                        state.error != "",
                        rx.box(
                            rx.text(f"Error: {state.error}", color="#ffd8e1",
                                    font_size="0.9rem"),
                            padding="0.8rem 1rem", border_radius="12px",
                            background="rgba(90,35,28,0.6)",
                            border="1px solid rgba(247,167,183,0.25)",
                            width="100%",
                        ),
                    ),

                    align="start", spacing="4", width="100%",
                    padding="1.4rem", border="1px solid rgba(255,255,255,0.12)",
                    border_radius="18px", background="rgba(255,255,255,0.02)",
                ),

                # ---- results panel ----
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.text("Forecast result", font_size="0.82rem",
                                    color="#c9a84c", font_weight="800",
                                    letter_spacing="0.12em", text_transform="uppercase"),
                            rx.heading("Load curve", font_size="1.5rem"),
                            align="start", spacing="1",
                        ),
                        rx.cond(
                            state.loading,
                            rx.text("Running…", color="#9b8f7c"),
                            rx.cond(
                                state.latency_ms is not None,
                                rx.text(f"{state.latency_ms} ms", color="#7ec8a8",
                                        font_weight="700"),
                                rx.text("Idle", color="#6f6554"),
                            ),
                        ),
                        justify="between", align="center", width="100%",
                    ),

                    # summary cards (text computed in go(), rendered as Vars)
                    rx.hstack(
                        stat_card("Region", state.summary_region),
                        stat_card("Points", state.summary_points),
                        stat_card("Peak (MW)", state.summary_peak),
                        spacing="3", width="100%",
                    ),

                    # the chart (data = real Figure stored in state by go())
                    rx.plotly(
                        data=ForecastState.chart_figure,
                        width="100%",
                        height="420px",
                    ),

                    # the table (only when there are rows)
                    rx.cond(
                        state.rows.length() > 0,
                        rx.scroll_area(
                            rx.table.root(
                                rx.table.header(
                                    rx.table.row(
                                        rx.table.column_header_cell("Datetime"),
                                        rx.table.column_header_cell("Predicted MW"),
                                    ),
                                ),
                                rx.table.body(
                                    rx.foreach(
                                        state.rows,
                                        lambda r: rx.table.row(
                                            rx.table.cell(r[0]),
                                            rx.table.cell(r[1]),
                                        ),
                                    ),
                                ),
                                variant="surface", size="1", width="100%",
                            ),
                            type="always", scrollbars="vertical", max_height="360px",
                            width="100%", border_radius="12px",
                        ),
                    ),

                    align="start", spacing="4", width="100%",
                    padding="1.4rem", border="1px solid rgba(255,255,255,0.12)",
                    border_radius="18px", background="rgba(255,255,255,0.02)",
                ),

                grid_template_columns="minmax(300px, 0.9fr) minmax(0, 1.6fr)",
                gap="1.2rem",
                width="100%",
            ),

            rx.text("Research-grade model. Not for operational grid decisions.",
                    color="#6f6554", font_size="0.85rem", padding_top="0.5rem"),
        ),
        max_width="1280px",
        padding="1.2rem 1rem 3rem",
        background="radial-gradient(circle at 14% 8%, rgba(201,168,76,0.10), transparent 26%), #0d0b09",
        min_width="100vw",
        min_height="100vh",
    )


def _peak_label(rows: List[List]):
    """'1234 MW at 15:00' — or '—' when empty. Helper used before figure render."""
    if not rows:
        return "—"
    peak = max(rows, key=lambda r: r[1])
    return f"{peak[1]:,.0f} MW @ {peak[0][-5:]}"


# Reflex app entry point
app = rx.App(theme=rx.theme(appearance="dark", accent_color="gold"))
app.add_page(page, route="/", title="Smart Grid Load Forecasting")
