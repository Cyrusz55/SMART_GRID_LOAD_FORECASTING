"""rxconfig.py — Reflex configuration for the smart-grid frontend.

Run from this folder (frontend/) with:
    reflex run

The app module name (app_name) is 'load_forecast' (the file load_forecast.py
contains `app = rx.App(...)`).
"""
import reflex as rx

config = rx.Config(
    app_name="load_forecast",
    # See: Reflex runs with cwd = this folder (frontend/), so a flat module
    # name resolves to load_forecast.py sitting next to rxconfig.py.
    app_module_import="load_forecast",
    # Frontend dev server bind address/port.
    frontend_port=3000,
    # Reflex's own backend (state sync / event websocket) lives here.
    # IMPORTANT: api_url below must point at THIS same port (8000) so the
    # frontend can open its /_event websocket. Your FastAPI app must run on a
    # DIFFERENT port (we use 8001) so the two backends don't collide.
    backend_port=8000,
    # Where the compiled web assets go. Kept inside this folder.
    api_url="http://127.0.0.1:8000",
    deploy_url="http://127.0.0.1:3000",
    # Frontend origins allowed to open the state/event websocket (/_event).
    # python-socketio rejects a websocket whose Origin header is NOT in this
    # list with HTTP 403. The browser sends Origin: http://localhost:3000 (or
    # http://127.0.0.1:3000), so both MUST be here explicitly. Do NOT rely on
    # the default ("*",) — socketio + credentials rejects it.
    cors_allowed_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
)
