# Hydrology — Water Balance Simulation

A compact Streamlit app for running a LAIÖ-style water-balance simulation with optional AI-assisted parameter tuning and results analysis.

Quick overview
- Purpose: simulate catchment water balance, inspect metrics and time-series, and optionally let an AI suggest model parameters.
- UI: sidebar for inputs (location, data source, parameters, API keys), main area for plots, metrics and CSV export.

Quickstart
- Requirements: Python 3.8+ and dependencies in `requirements.txt`.
- Install and run (recommended):
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```
- Quick fallback if `requirements.txt` is missing:
```bash
pip install streamlit pandas numpy plotly requests
```

Run
```bash
streamlit run app.py
```

Features
- Fetch weather: geocode by place name (Open‑Meteo) or fetch by coordinates (Open‑Meteo archive).
- Data sources: fetched weather, uploaded CSV, or generated example dataset.
- Sidebar preview: small time-series preview shown after a successful fetch.
- Parameter control: edit model parameters in the sidebar; supports Apply / Undo and maintains a parameter history.
- AI Auto‑Tune: request parameter suggestions from TAMU (default), OpenAI, or Gemini. Suggestions can be previewed, applied, or undone. Mock mode supported for local testing.
- Run simulation: computes water‑balance time series and summary metrics, displays interactive Plotly charts, and exposes raw results for CSV download.
- Persistence: last results and AI summary persist in Streamlit session state during a session.

Usage (high level)
- Sidebar: enter location or coordinates, choose data source, upload CSV if needed, and set model parameters or enable Auto‑Tune.
- AI: enter provider API key (sidebar) or store in `.streamlit/secrets.toml`. Use "Test TAMU API" to validate connectivity when using TAMU.
- Run: click "Run Simulation" to produce metrics, interactive charts, the results table, and a CSV download link.

AI integration notes
- Supported providers: TAMU (default), OpenAI, Gemini. Provide per-provider API keys in the sidebar or via Streamlit secrets.
- TAMU specifics: base URL can be overridden, model list helper exists (`tamu_list_models.py`), and mock mode exercises UI flows without real calls.

Developer notes
- Main files:
  - `app.py` — Streamlit UI, AI helpers, and session-state management
  - `model.py` — water-balance simulation routines
  - `tamu_list_models.py` — convenience script to list TAMU models (requires TAMU key)
- Important session-state keys: `rain_df`, parameters (`sh, sw, sstar, sfc, n, zr, ks, ew, emax, beta, s0`), `tamu_suggestions`, `param_history`, `last_results`, `ai_summary`.

Troubleshooting
- AI errors: check API key, network/DNS, and use Mock TAMU AI to validate UI behavior.
- TAMU DNS issues: test resolution with `dig` or `python -c "import socket; print(socket.gethostbyname('api.tamu.ai'))"`.
- Logs: AI call traces may be written to `ai_logs.txt` if enabled by the app.

Security
- Do not commit API keys. Use `.streamlit/secrets.toml` or environment variables for deployments and platform secret managers for cloud deploys.
