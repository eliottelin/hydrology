import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
import json
import socket
from model import water_balance_simulation
from datetime import date, datetime, timedelta

# --- AI helpers ---
def parse_ai_json(text: str):
    """Extract JSON object from text robustly.
    Looks for JSON_START/JSON_END markers first, then falls back to finding the first balanced {...}.
    Returns a dict or None.
    """
    if not text or not isinstance(text, str):
        return None
    # prefer explicit markers
    if "JSON_START" in text and "JSON_END" in text:
        try:
            start = text.index("JSON_START") + len("JSON_START")
            end = text.index("JSON_END", start)
            candidate = text[start:end].strip()
            return json.loads(candidate)
        except Exception:
            pass
    # strip code fences
    if "```" in text:
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
    # find first balanced {...}
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start : i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None





def call_tamu_api(api_key: str, prompt: str, model: str = "protected.gemini-2.5-flash-lite", base_url: str = "https://chat-api.tamu.ai", mock: bool = False, timeout: int = 30):
    """Call the TAMU AI chat completions endpoint and return assistant text or an error string prefixed with __error__.
    Uses POST {base_url.rstrip('/')}/api/chat/completions with payload matching docs.
    If mock=True returns a canned response for local dev.
    """
    if mock:
        return '{"message":"This is a mocked TAMU AI response."}\nJSON_START\n{"sh":0.08,"sw":0.11,"sstar":0.33,"sfc":0.40,"n":0.55,"zr":500.0,"ks":200.0,"ew":0.1,"emax":5.0,"beta":3.0,"s0":0.3}\nJSON_END'

    if not api_key:
        return "__error__:NoAPIKey:No TAMU API key provided"

    url = f"{base_url.rstrip('/')}/api/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "stream": False,
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            return f"__error__:HTTP_{r.status_code}:{r.text}"
        data = r.json()
        # docs show response similar to OpenAI: find assistant content
        # Support both choices[*].message.content and top-level text
        choices = data.get("choices") or []
        if choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message") or first.get("text") or {}
                if isinstance(msg, dict):
                    return msg.get("content") or json.dumps(msg)
                return msg
        # fallback: try top-level fields
        if "text" in data:
            return data.get("text")
        return json.dumps(data)
    except requests.exceptions.RequestException as e:
        return f"__error__:{e.__class__.__name__}:{str(e)}"
    except Exception as e:
        try:
            import traceback

            return f"__error__:{traceback.format_exc()}"
        except Exception:
            return "__error__:unknown"


@st.cache_data(ttl=3600)
def fetch_tamu_models(api_key: str, base_url: str = "https://chat-api.tamu.ai", mock: bool = False):
    """Return a list of model ids from the TAMU /openai/models endpoint. Returns empty list on failure.
    Cached for 1 hour.
    """
    if mock:
        return ["protected.gemini-2.5-flash-lite"]
    if not api_key:
        return []
    try:
        url = f"{base_url.rstrip('/')}/openai/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code != 200:
            return []
        j = r.json()
        # response can be {'data': [...]} or a list
        data = j.get('data') if isinstance(j, dict) else j
        if data is None:
            data = []
        models = []
        for item in data:
            if isinstance(item, dict):
                mid = item.get('id') or item.get('model') or item.get('name') or item.get('modelId')
                if mid:
                    models.append(mid)
        # deduplicate
        seen = []
        out = []
        for m in models:
            if m not in seen:
                seen.append(m)
                out.append(m)
        return out
    except Exception:
        return []


def show_ai_summary_block(results, button_key: str = "generate_ai_summary"):
    """Render AI summary controls for provided results DataFrame. Stores summary in session_state['ai_summary'].
    This is safe to call multiple times; uses a stable button key so it persists across reruns.
    """
    if results is None:
        return
    st.markdown("---")
    with st.expander("AI: Generate summary and structured JSON", expanded=False):
        st.write("Generate a concise summary and a small JSON payload with metrics. Requires TAMU API key (or mock enabled).")
        if st.button("Generate AI Summary (include model results)", key=button_key):
            token_to_use = st.session_state.get('tamu_api_key', tamu_api_key)
            with st.spinner("Contacting TAMU AI for a summary..."):
                try:
                    total_rain = float(results['Rain'].sum())
                    total_et = float(results['et'].sum())
                    total_runoff = float(results['q'].sum())
                    mean_soil = float(results['s'].mean())
                except Exception:
                    total_rain = total_et = total_runoff = mean_soil = 0.0
                prompt = f"""
Provide a short, user-facing summary (3-5 sentences) of the simulation results for location {st.session_state.get('location_name','(unknown)')} (Lat:{st.session_state.get('loc_lat','?')}, Lon:{st.session_state.get('loc_lon','?')}).
Include the following numeric results: total_rain={total_rain:.2f} mm, total_et={total_et:.2f} mm, total_runoff={total_runoff:.2f} mm, mean_soil={mean_soil:.3f}.
Return two things: (1) a short plain-text summary, and (2) a JSON object exactly between markers JSON_START and JSON_END containing keys: 'total_rain','total_et','total_runoff','mean_soil','recommendation'.
Do not include extra commentary outside the requested text and JSON.
"""
                resp = call_tamu_api(token_to_use or "", prompt, model=effective_tamu_model, base_url=st.session_state.get('tamu_base_url', tamu_base_url), mock=st.session_state.get('tamu_mock', tamu_mock))
                if not resp:
                    st.error("No response from TAMU AI. Check API key and network connectivity.")
                elif isinstance(resp, str) and resp.startswith("__error__"):
                    st.error("AI request failed (network or DNS). See raw error below.")
                    with st.expander("Raw AI error"):
                        st.code(resp)
                else:
                    st.session_state['ai_summary'] = resp
                    parsed = parse_ai_json(resp or "")
                    if parsed:
                        st.session_state['ai_summary_json'] = parsed
                    with st.expander("AI raw response", expanded=True):
                        st.code(resp or "(no response)")
        # show any previously generated summary
        if 'ai_summary' in st.session_state:
            st.markdown("**Last AI summary (raw)**")
            with st.expander("Show last AI summary", expanded=False):
                st.code(st.session_state.get('ai_summary'))
            if 'ai_summary_json' in st.session_state:
                st.markdown("**Last AI structured JSON**")
                st.json(st.session_state.get('ai_summary_json'))


def get_precipitation(lat: float, lon: float, start_date, end_date) -> pd.DataFrame:
    """Fetch daily precipitation_sum from Open-Meteo archive API and return DataFrame with Date, Rain."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": float(lat),
        "longitude": float(lon),
        "start_date": str(start_date),
        "end_date": str(end_date),
        "daily": "precipitation_sum",
        "timezone": "auto",
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    j = r.json()
    if "daily" in j and "time" in j["daily"]:
        df = pd.DataFrame({"Date": j["daily"]["time"], "Rain": j["daily"]["precipitation_sum"]})
        df["Date"] = pd.to_datetime(df["Date"])
        return df
    return pd.DataFrame({"Date": [], "Rain": []})

# --- Page Config ---
st.set_page_config(page_title="Hydrology Lab Model", layout="wide")

st.title("🌊 Hydrology Lab: Water Balance Simulation")
st.markdown("This app simulates soil moisture dynamics based on the **Laio et al. (2001)** model.")

# --- SIDEBAR: Location (simple Open-Meteo geocoding) ---
def get_geocoding_results(query: str):
    """Return first geocoding result from Open-Meteo or None."""
    if not query:
        return None
    try:
        url = "https://geocoding-api.open-meteo.com/v1/search"
        r = requests.get(url, params={"name": query, "count": 5}, timeout=8)
        r.raise_for_status()
        data = r.json()
        results = data.get("results") or []
        if results:
            return results[0]
    except Exception:
        return None
    return None


st.sidebar.header("Location")
place = st.sidebar.text_input("Place name (e.g. Austin, TX)", value="")
if st.sidebar.button("Lookup place") and place:
    with st.sidebar.spinner("Looking up place..."):
        geo = get_geocoding_results(place)
        if not geo:
            st.sidebar.error("No results found")
        else:
            st.sidebar.success(f"Found: {geo.get('name')}, {geo.get('country')}")
            st.session_state['loc_lat'] = geo.get('latitude')
            st.session_state['loc_lon'] = geo.get('longitude')
            st.session_state['location_name'] = f"{geo.get('name')}, {geo.get('country')}"
            # One-click: automatically fetch precipitation for found place (last 365 days)
            try:
                sd = datetime.today().date() - timedelta(days=365)
                ed = datetime.today().date()
                df_precip = get_precipitation(geo.get('latitude'), geo.get('longitude'), sd, ed)
                if df_precip is not None and not df_precip.empty:
                    st.session_state['rain_df'] = df_precip
                    st.sidebar.success(f"Fetched {len(df_precip)} days for {geo.get('name')}")
                    # mini preview plot (last 30 days)
                    try:
                        mini_fig = px.bar(df_precip.tail(30), x='Date', y='Rain', labels={'Rain': 'mm'}, title='Last 30 days (preview)')
                        st.sidebar.plotly_chart(mini_fig, width=260)
                    except Exception:
                        pass
                else:
                    st.sidebar.info("Lookup succeeded but no precipitation data available for that period.")
            except Exception as e:
                st.sidebar.error(f"Auto-fetch failed: {e}")

# show selected coords (if any)
if 'loc_lat' in st.session_state and 'loc_lon' in st.session_state:
    st.sidebar.markdown(f"**Selected:** {st.session_state.get('location_name','')}  ")
    st.sidebar.markdown(f"Lat: {st.session_state['loc_lat']:.5f}, Lon: {st.session_state['loc_lon']:.5f}")

# Coordinate fetch controls
with st.sidebar.expander("Fetch weather by coordinates", expanded=False):
    lat_input = st.number_input("Latitude", value=float(st.session_state.get('loc_lat', 30.6280)), format="%.6f")
    lon_input = st.number_input("Longitude", value=float(st.session_state.get('loc_lon', -96.3344)), format="%.6f")
    start_date = st.date_input("Start date", datetime.today().date() - timedelta(days=365))
    end_date = st.date_input("End date", datetime.today().date())
    if st.button("Fetch weather for coordinates"):
        try:
            df_precip = get_precipitation(lat_input, lon_input, start_date, end_date)
            if df_precip is not None and not df_precip.empty:
                st.session_state['rain_df'] = df_precip
                st.success(f"Loaded {len(df_precip)} days from Open-Meteo")
                # also set session location name and coords
                st.session_state['loc_lat'] = float(lat_input)
                st.session_state['loc_lon'] = float(lon_input)
                st.session_state['location_name'] = f"Coords: {lat_input:.4f}, {lon_input:.4f}"
                # mini preview plot (last 30 days)
                try:
                    mini_fig = px.bar(df_precip.tail(30), x='Date', y='Rain', labels={'Rain': 'mm'}, title='Last 30 days (preview)')
                    st.plotly_chart(mini_fig, width=260)
                except Exception:
                    pass
            else:
                st.error("No precipitation data returned")
        except Exception as e:
            st.error(f"Weather fetch failed: {e}")

# --- TAMU AI settings (per-user token supported) ---
st.sidebar.header("TAMU AI")
tamu_api_key = st.sidebar.text_input("TAMU AI API Key (optional)", type="password")
tamu_base_url = st.sidebar.text_input("TAMU API base URL", value="https://chat-api.tamu.ai")
tamu_mock = st.sidebar.checkbox("Mock TAMU AI (dev)", value=False)

# mirror into session_state so helper functions can read latest values
st.session_state['tamu_api_key'] = tamu_api_key
st.session_state['tamu_base_url'] = tamu_base_url
st.session_state['tamu_mock'] = tamu_mock

# Try to fetch available TAMU models and present as a dropdown when possible
available_models = []
if tamu_api_key and not tamu_mock:
    try:
        with st.sidebar.spinner("Fetching TAMU models..."):
            available_models = fetch_tamu_models(tamu_api_key, tamu_base_url, tamu_mock)
    except Exception:
        available_models = []

if available_models:
    # use session_state key so selection persists
    tamu_model = st.sidebar.selectbox("TAMU Model", options=available_models, index=0, key='tamu_model')
else:
    tamu_model = st.sidebar.text_input("TAMU Model (enter id)", value=st.session_state.get('tamu_model', "protected.gemini-2.5-flash-lite"), key='tamu_model')

# normalize effective model string for downstream calls
effective_tamu_model = str(st.session_state.get('tamu_model') or tamu_model or "protected.gemini-2.5-flash-lite")

if st.sidebar.button("Test TAMU API"):
    if not tamu_base_url:
        st.sidebar.error("Set the TAMU API base URL first.")
    else:
        # DNS preflight
        try:
            from urllib.parse import urlparse

            parsed = urlparse(tamu_base_url)
            host = parsed.hostname or tamu_base_url
            try:
                ip = socket.gethostbyname(host)
                st.sidebar.success(f"Resolved host {host} -> {ip}")
            except Exception as dns_e:
                st.sidebar.warning(f"Could not resolve host {host}: {dns_e}")
        except Exception:
            host = None

        try:
            test_prompt = "Test connection from Hydrology app. Reply with short text 'OK'."
            test_resp = call_tamu_api(tamu_api_key or "", test_prompt, model=effective_tamu_model, base_url=tamu_base_url, mock=tamu_mock)
            if not test_resp:
                st.sidebar.error("No response from TAMU API. Check network and base URL.")
            elif isinstance(test_resp, str) and test_resp.startswith("__error__"):
                st.sidebar.error("TAMU API call failed. See raw details below.")
                with st.sidebar.expander("TAMU raw error"):
                    st.code(test_resp)
            else:
                st.sidebar.success("TAMU API reachable (response shown below).")
                with st.sidebar.expander("TAMU test response"):
                    st.code(test_resp)
        except Exception as e:
            st.sidebar.error(f"TAMU test failed: {e}")


# parameter keys (used by auto-tune prompt)
param_keys = ['sh','sw','sstar','sfc','n','zr','ks','ew','emax','beta','s0']

# Auto-tune removed. AI features have been disabled per user request.

# --- SIDEBAR: PARAMETERS ---

# Use session_state-backed widgets for parameters so they can be updated programmatically
st.sidebar.subheader("Model Parameters")
st.sidebar.write("(these can be auto-tuned by the AI)")
st.sidebar.number_input("Hygroscopic Point (sh)", 0.0, 1.0, 0.08, format="%.2f", key='sh')
st.sidebar.number_input("Wilting Point (sw)", 0.0, 1.0, 0.11, format="%.2f", key='sw')
st.sidebar.number_input("Stomatal Closure (s*)", 0.0, 1.0, 0.33, format="%.2f", key='sstar')
st.sidebar.number_input("Field Capacity (sfc)", 0.0, 1.0, 0.40, format="%.2f", key='sfc')
st.sidebar.number_input("Porosity (n)", 0.1, 1.0, 0.55, format="%.2f", key='n')
st.sidebar.number_input("Root Depth (Zr) [mm]", 10.0, 2000.0, 500.0, key='zr')
st.sidebar.number_input("Saturated Conductivity (Ks) [mm/day]", 0.0, 1000.0, 200.0, key='ks')
st.sidebar.number_input("Evap at Wilting (Ew) [mm/day]", 0.0, 10.0, 0.1, key='ew')
st.sidebar.number_input("Max Evap (Emax) [mm/day]", 0.0, 20.0, 5.0, key='emax')
st.sidebar.number_input("Leakage Parameter (Beta)", 0.0, 20.0, 3.0, key='beta')
st.sidebar.slider("Initial Soil Moisture (s0)", 0.0, 1.0, 0.3, key='s0')

# Build parameter dict from session_state
p = {k: st.session_state.get(k) for k in param_keys}

# Auto-tune (TAMU) control
with st.sidebar.container():
    st.markdown("---")
    st.write("AI: Auto-tune model parameters")
    if st.button("Auto-Tune parameters (TAMU)", key="auto_tune"):
        # Build a prompt that includes current parameter values and recent metrics
        last = st.session_state.get('last_results')
        try:
            lr = float(last['Rain'].sum()) if last is not None else 0.0
            let = float(last['et'].sum()) if last is not None else 0.0
            lq = float(last['q'].sum()) if last is not None else 0.0
            ls = float(last['s'].mean()) if last is not None else 0.0
            sample_days = int(len(last)) if last is not None else 0
        except Exception:
            lr = let = lq = ls = 0.0
            sample_days = 0

        prompt = (
            "You are an expert hydrologist. Suggest tuned model parameters for the Laio et al. water-balance model. "
            f"Current parameter values: {json.dumps(p)}. "
            f"Recent metrics over {sample_days} days: total_rain={lr:.2f} mm, total_et={let:.2f} mm, total_runoff={lq:.2f} mm, mean_soil={ls:.3f}. "
            "Return a JSON object between markers JSON_START and JSON_END with numeric values for keys: "
            "'sh','sw','sstar','sfc','n','zr','ks','ew','emax','beta','s0'. Do not include extra text."
        )

        resp = call_tamu_api(tamu_api_key or "", prompt, model=effective_tamu_model, base_url=tamu_base_url, mock=tamu_mock)
        if not resp:
            st.error("No response from TAMU API. Check your API key and network.")
        elif isinstance(resp, str) and resp.startswith("__error__"):
            st.error("TAMU API call failed. See raw details below.")
            with st.expander("TAMU raw error"):
                st.code(resp)
        else:
            parsed = parse_ai_json(resp or "")
            if not parsed or not isinstance(parsed, dict):
                st.error("TAMU did not return parsable JSON. See raw response.")
                with st.expander("TAMU raw response"):
                    st.code(resp)
            else:
                # Show suggestions and offer Apply / Undo
                st.sidebar.success("Received suggested parameters from TAMU")
                with st.sidebar.expander("Suggested parameters (TAMU)", expanded=True):
                    st.json(parsed)
                    if st.button("Apply suggested parameters", key="apply_tamu_params"):
                        # Save current params for undo
                        hist = st.session_state.setdefault('param_history', [])
                        hist.append({k: st.session_state.get(k) for k in param_keys})
                        # Apply suggested values where present
                        for k, v in parsed.items():
                            if k in param_keys:
                                try:
                                    st.session_state[k] = float(v)
                                except Exception:
                                    # ignore non-numeric
                                    pass
                        st.success("Applied suggested parameters")
                    if st.button("Undo last parameter apply", key="undo_tamu_params"):
                        hist = st.session_state.get('param_history', [])
                        if hist:
                            last_params = hist.pop()
                            for k, v in last_params.items():
                                st.session_state[k] = v
                            st.success("Reverted to previous parameters")
                        else:
                            st.info("No previous parameter snapshot available to undo.")

# --- WEATHER DATA ---
st.header("Step 1: Weather Data")

# Create a default dummy dataset (so the app isn't empty)
dates = pd.date_range(start="2024-01-01", periods=100)
# Create synthetic rain: mostly zeros, some random storms
np.random.seed(42)
rain_vals = np.random.exponential(scale=5, size=100) * (np.random.rand(100) > 0.7)
default_df = pd.DataFrame({'Date': dates, 'Rain': rain_vals})

data_source = st.radio("Choose Data Source:", ["Use Fetched Weather", "Upload CSV", "Generated Example Data"])

# Use rain_df from session_state if it was previously fetched/uploaded
rain_df = st.session_state.get('rain_df', None)

if data_source == "Use Fetched Weather":
    if rain_df is None or rain_df.empty:
        st.warning("No fetched weather data available. Fetch by coordinates or lookup a place first.")
        # fall back to example only for continuity
        rain_df = default_df
    else:
        st.success(f"Using fetched weather ({len(rain_df)} days)")

elif data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV (Columns: 'Date', 'Rain')")
    if uploaded_file:
        rain_df = pd.read_csv(uploaded_file)
        # store uploaded into session
        st.session_state['rain_df'] = rain_df
    
        # 1. Standardize Column Names (Handle "PRCP" or "Rain")
        # If the file has 'PRCP', rename it to 'Rain'
        if 'PRCP' in rain_df.columns:
            rain_df = rain_df.rename(columns={'PRCP': 'Rain'})
    
        # Same for Date (Handle 'DATE' or 'Date')
        if 'DATE' in rain_df.columns:
            rain_df = rain_df.rename(columns={'DATE': 'Date'})

        # 2. Ask User about Units
        is_inches = st.checkbox("Convert Inches to MM? (Check this for NOAA data)")
        if is_inches:
            rain_df['Rain'] = rain_df['Rain'] * 25.4
elif data_source == "Generated Example Data":
    # user explicitly asked for the generated example dataset
    rain_df = default_df
    st.session_state['rain_df'] = default_df
    st.info("Using generated example data (100 days).")

# --- SIMULATION ---
if st.button("Run Simulation", type="primary"):
    if rain_df is not None:
        # Ensure 'Date' is datetime
        try:
            if 'Date' in rain_df.columns:
                rain_df['Date'] = pd.to_datetime(rain_df['Date'])
            
            # --- RUN MODEL ---
            results = water_balance_simulation(p, rain_df)
            
            # --- PLOTTING ---
            st.header("Step 2: Results")

            # Summaries / Metrics (with units)
            total_rain = results['Rain'].sum()
            mean_rain = results['Rain'].mean()
            total_interception = results['i'].sum()
            total_runoff = results['q'].sum()
            mean_soil = results['s'].mean()
            total_et = results['et'].sum()
            total_leakage = results['l'].sum()

            cols = st.columns(6)
            cols[0].metric("Total Rain", f"{total_rain:.1f} mm")
            cols[1].metric("Mean Rain/day", f"{mean_rain:.2f} mm/d")
            cols[2].metric("Total Interception", f"{total_interception:.1f} mm")
            cols[3].metric("Total Runoff", f"{total_runoff:.1f} mm")
            cols[4].metric("Total ET", f"{total_et:.1f} mm")
            cols[5].metric("Mean Soil Moisture", f"{mean_soil:.2f}")

            # persist results so AI summary can reference them across reruns
            try:
                st.session_state['last_results'] = results
                st.session_state['last_results_ts'] = datetime.now().isoformat()
            except Exception:
                # if session_state can't store the DataFrame for any reason, ignore
                pass

            st.markdown("---")

            # 1) Precipitation (inputs) — Rain & Canopy Interception
            fig_rain_ci = px.bar(results, x='Date', y='Rain', labels={'Rain': 'Precipitation (mm)'}, title="Precipitation & Canopy Interception (mm)")
            fig_rain_ci.add_trace(go.Scatter(x=results['Date'], y=results['i'], mode='lines', name='Interception (CI) (mm)', line=dict(color='orange')))
            fig_rain_ci.update_yaxes(title_text='Precipitation (mm)')
            st.plotly_chart(fig_rain_ci, width='stretch')

            # 2) Soil moisture (fraction)
            fig_s = px.line(results, x='Date', y='s', title="Soil Moisture (s) — fraction (0-1)")
            fig_s.add_hline(y=p['sw'], line_dash="dash", line_color="red", annotation_text="Wilting Point")
            fig_s.add_hline(y=p['sfc'], line_dash="dash", line_color="green", annotation_text="Field Capacity")
            fig_s.update_yaxes(title_text='Soil moisture (fraction)')
            st.plotly_chart(fig_s, width='stretch')

            # 3) ET, Leakage, and Runoff combined (all mm)
            fig_elq = go.Figure()
            fig_elq.add_trace(go.Bar(x=results['Date'], y=results['q'], name='Runoff (q) (mm)', marker_color='lightskyblue', opacity=0.6))
            fig_elq.add_trace(go.Scatter(x=results['Date'], y=results['et'], name='ET (mm)', line=dict(color='green')))
            fig_elq.add_trace(go.Scatter(x=results['Date'], y=results['l'], name='Leakage (L) (mm)', line=dict(color='brown')))
            fig_elq.update_layout(title='ET, Leakage, and Runoff (mm)', yaxis_title='mm')
            st.plotly_chart(fig_elq, width='stretch')
            
            # 5. Data Table
            with st.expander("View Raw Data"):
                st.dataframe(results)
                
            # 6. Download
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "simulation_results.csv", "text/csv")

            # TAMU AI summary (optional)
            if tamu_api_key or tamu_mock:
                token_to_use = tamu_api_key
                if st.button("Generate AI Summary (include model results)"):
                    with st.spinner("Contacting TAMU AI for a summary..."):
                        prompt = f"""
Provide a short, user-facing summary (3-5 sentences) of the simulation results for location {st.session_state.get('location_name','(unknown)')} (Lat:{st.session_state.get('loc_lat','?')}, Lon:{st.session_state.get('loc_lon','?')}).
Include the following numeric results: total_rain={total_rain:.2f} mm, total_et={total_et:.2f} mm, total_runoff={total_runoff:.2f} mm, mean_soil={mean_soil:.3f}.
Return two things: (1) a short plain-text summary, and (2) a JSON object exactly between markers JSON_START and JSON_END containing keys: 'total_rain','total_et','total_runoff','mean_soil','recommendation'.
Example JSON:
JSON_START
{
  "total_rain": {total_rain:.2f},
  "total_et": {total_et:.2f},
  "total_runoff": {total_runoff:.2f},
  "mean_soil": {mean_soil:.3f},
  "recommendation": "short recommendation"
}
JSON_END
Do not include extra commentary outside the requested text and JSON.
"""
                        if tamu_mock:
                            resp = '{"message":"This is a mocked TAMU response."}\nJSON_START\n{"total_rain":' + f'{total_rain:.2f}' + ',"total_et":' + f'{total_et:.2f}' + ',"total_runoff":' + f'{total_runoff:.2f}' + ',"mean_soil":' + f'{mean_soil:.3f}' + ',"recommendation":"mocked suggestion"}\nJSON_END'
                        else:
                            resp = call_tamu_api(token_to_use or "", prompt, model=effective_tamu_model, base_url=tamu_base_url, mock=tamu_mock)
                    # log and show
                    if not resp:
                        st.error("No response from TAMU AI. Check API key and network connectivity.")
                        with st.expander("Raw AI response"):
                            st.code(str(resp) or "(no response)")
                    elif isinstance(resp, str) and resp.startswith("__error__"):
                        st.error("AI request failed (network or DNS). Check your internet connection and the TAMU API host.")
                        with st.expander("Raw AI response (error)"):
                            st.code(resp)
                    else:
                        parsed_summary = parse_ai_json(resp or "")
                        with st.expander("AI raw response"):
                            st.code(resp or "(no response)")
                        if parsed_summary:
                            st.markdown("### AI Summary")
                            st.info(resp.split('JSON_START')[0].strip() if 'JSON_START' in (resp or '') else (resp or ""))
                            st.subheader("AI structured summary")
                            st.json(parsed_summary)
                            st.session_state['ai_summary'] = resp
                            st.session_state['ai_summary_json'] = parsed_summary
                        else:
                            st.error("AI did not return structured JSON. See raw response in the expander.")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Check your CSV columns. They must be 'Date' and 'Rain'.")

# If there are persisted results from a previous run, render them so summaries persist across reruns
if 'last_results' in st.session_state:
    try:
        results = st.session_state['last_results']
        st.header("Step 2: Results")

        total_rain = results['Rain'].sum()
        mean_rain = results['Rain'].mean()
        total_interception = results['i'].sum()
        total_runoff = results['q'].sum()
        mean_soil = results['s'].mean()
        total_et = results['et'].sum()
        total_leakage = results['l'].sum()

        cols = st.columns(6)
        cols[0].metric("Total Rain", f"{total_rain:.1f} mm")
        cols[1].metric("Mean Rain/day", f"{mean_rain:.2f} mm/d")
        cols[2].metric("Total Interception", f"{total_interception:.1f} mm")
        cols[3].metric("Total Runoff", f"{total_runoff:.1f} mm")
        cols[4].metric("Total ET", f"{total_et:.1f} mm")
        cols[5].metric("Mean Soil Moisture", f"{mean_soil:.2f}")

        st.markdown("---")

        fig_rain_ci = px.bar(results, x='Date', y='Rain', labels={'Rain': 'Precipitation (mm)'}, title="Precipitation & Canopy Interception (mm)")
        fig_rain_ci.add_trace(go.Scatter(x=results['Date'], y=results['i'], mode='lines', name='Interception (CI) (mm)', line=dict(color='orange')))
        fig_rain_ci.update_yaxes(title_text='Precipitation (mm)')
        st.plotly_chart(fig_rain_ci, width='stretch')

        fig_s = px.line(results, x='Date', y='s', title="Soil Moisture (s) — fraction (0-1)")
        fig_s.add_hline(y=p['sw'], line_dash="dash", line_color="red", annotation_text="Wilting Point")
        fig_s.add_hline(y=p['sfc'], line_dash="dash", line_color="green", annotation_text="Field Capacity")
        fig_s.update_yaxes(title_text='Soil moisture (fraction)')
        st.plotly_chart(fig_s, width='stretch')

        fig_elq = go.Figure()
        fig_elq.add_trace(go.Bar(x=results['Date'], y=results['q'], name='Runoff (q) (mm)', marker_color='lightskyblue', opacity=0.6))
        fig_elq.add_trace(go.Scatter(x=results['Date'], y=results['et'], name='ET (mm)', line=dict(color='green')))
        fig_elq.add_trace(go.Scatter(x=results['Date'], y=results['l'], name='Leakage (L) (mm)', line=dict(color='brown')))
        fig_elq.update_layout(title='ET, Leakage, and Runoff (mm)', yaxis_title='mm')
        st.plotly_chart(fig_elq, width='stretch')

        with st.expander("View Raw Data"):
            st.dataframe(results)

        csv = results.to_csv(index=False).encode('utf-8')
        st.download_button("Download Results", csv, "simulation_results.csv", "text/csv")

        # Render the unified AI summary block for persisted results
        try:
            show_ai_summary_block(results, button_key="generate_ai_summary_persisted")
        except Exception:
            # don't fail page rendering if summary block has issues
            pass
    except Exception:
        # if anything goes wrong rendering persisted results, skip gracefully
        pass