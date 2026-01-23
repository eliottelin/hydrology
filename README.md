# Hydrology Lab — Water Balance Simulation

Small Streamlit app to run a Laio et al. water-balance simulation with optional AI-assisted parameter tuning (TAMU AI).

## Features
- Fetch precipitation by place name (Open-Meteo geocoding) or coordinates (Open-Meteo archive).
- Mini preview plot in the sidebar after a successful fetch.
- Data source selection: Use fetched weather / Upload CSV / Generated example.
- Model parameters editable in sidebar; AI Auto‑Tune (TAMU) can suggest and apply parameters (Apply / Undo).
- Run simulation -> metrics, charts, raw table, and CSV download.
- Optional TAMU AI integration:
  - Per-user API key (entered in sidebar).
  - Model dropdown (fetched from TAMU when key present) or manual model id input.
  - Mock mode for local dev.
  - Test TAMU API button (DNS preflight + request).
- Persistent last results and AI summary stored in session state.
- Local safe-mode summarizer used if TAMU not configured / mocked.

## Quick start (dev)
1. Create virtualenv and install deps:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
   If there is no `requirements.txt`, install:
   ```bash
   pip install streamlit pandas numpy plotly requests
   ```

2. Run the app:
   ```bash
   streamlit run app.py
   ```

3. Sidebar workflow:
   - Location -> look up place or expand "Fetch weather by coordinates".
   - After fetch, mini preview will appear in sidebar.
   - Choose Data Source on main page.
   - Adjust model parameters (or use Auto‑Tune).
   - Click "Run Simulation".
   - View metrics (top), summary (if generated), and charts below.

4. TAMU AI (optional):
   - Enter your TAMU API key in the sidebar (recommended to put secrets in `.streamlit/secrets.toml` for deployments).
   - Set TAMU API base URL if different (default `https://chat-api.tamu.ai`).
   - Toggle Mock TAMU AI for development.
   - Click "Test TAMU API" to validate connectivity.
   - Click "Auto-Tune parameters (TAMU)" to request suggestions.
     - Suggestions are shown in an expander.
     - Click "Apply suggested parameters" to immediately apply.
     - Click "Undo last parameter apply" to revert.

5. Troubleshooting
   - DNS / Name resolution errors for TAMU:
     - Check DNS: `dig api.tamu.ai +short` or `python -c "import socket; print(socket.gethostbyname('api.tamu.ai'))"`.
     - Check network / proxy settings.
   - If TAMU fails, enable `Mock TAMU AI (dev)` to test UI flows.
   - Logs (AI request traces) are appended to `ai_logs.txt` in project root (if enabled).

6. Developer notes
   - `tamu_list_models.py` can list available TAMU models (run locally with your API key).
   - Session state keys used: `rain_df`, parameter keys (`sh,sw,sstar,sfc,n,zr,ks,ew,emax,beta,s0`), `tamu_suggestions`, `param_history`, `last_results`, `ai_summary`.
   - UI widgets are session-state backed so parameter Apply/Undo persists across reruns.

## File overview
- `app.py` — main Streamlit app
- `model.py` — water-balance simulation logic
- `tamu_list_models.py` — (utility) list TAMU models (requires TAMU key)
- `ai_logs.txt` — appended AI call logs (project root)

## Security
- Do not commit API keys. Use `.streamlit/secrets.toml` or environment variables for deployments.