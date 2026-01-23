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

# Default TAMU model id used across the app
DEFAULT_TAMU_MODEL = "protected.claude-haiku-5.4"

# Initialize default model in session_state if not already set so the UI shows the intended default
if 'tamu_model' not in st.session_state:
    st.session_state['tamu_model'] = DEFAULT_TAMU_MODEL

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


# Parameter bounds used to validate/clamp suggested values
PARAM_BOUNDS = {
    'sh': (0.0, 1.0),
    'sw': (0.0, 1.0),
    'sstar': (0.0, 1.0),
    'sfc': (0.0, 1.0),
    'n': (0.1, 1.0),
    'zr': (10.0, 2000.0),
    'ks': (0.0, 1000.0),
    'ew': (0.0, 10.0),
    'emax': (0.0, 20.0),
    'beta': (0.0, 20.0),
    's0': (0.0, 1.0),
}


def clamp_param_value(key, value):
    """Clamp a numeric parameter value into allowed bounds where defined."""
    try:
        v = float(value)
    except Exception:
        return value
    if key in PARAM_BOUNDS:
        lo, hi = PARAM_BOUNDS[key]
        if v < lo:
            return lo
        if v > hi:
            return hi
    return v





def call_tamu_api(api_key: str, prompt: str, model: str = DEFAULT_TAMU_MODEL, base_url: str = "https://chat-api.tamu.ai", mock: bool = False, timeout: int = 30):
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


def generate_and_store_summary(results_df):
    """Generate AI summary for a given results DataFrame and store into session_state.
    Uses session_state values for token/model/base_url/mock. Returns the raw response or None.
    """
    if results_df is None:
        return None
    token = st.session_state.get('tamu_api_key')
    mock_mode = bool(st.session_state.get('tamu_mock', False))
    model_to_use = str(st.session_state.get('tamu_model') or DEFAULT_TAMU_MODEL)
    base_to_use = str(st.session_state.get('tamu_base_url') or "https://chat-api.tamu.ai")
    try:
        tr = float(results_df['Rain'].sum())
        te = float(results_df['et'].sum())
        tq = float(results_df['q'].sum())
        ms = float(results_df['s'].mean())
    except Exception:
        tr = te = tq = ms = 0.0
    prompt = f"""
Provide a short, user-facing summary (3-5 sentences) of the simulation results for location {st.session_state.get('location_name','(unknown)')} (Lat:{st.session_state.get('loc_lat','?')}, Lon:{st.session_state.get('loc_lon','?')}).
Include the following numeric results: total_rain={tr:.2f} mm, total_et={te:.2f} mm, total_runoff={tq:.2f} mm, mean_soil={ms:.3f}.
Return two things: (1) a short plain-text summary, and (2) a JSON object exactly between markers JSON_START and JSON_END containing keys: 'total_rain','total_et','total_runoff','mean_soil','recommendation'.
Do not include extra commentary outside the requested text and JSON.
"""
    resp = call_tamu_api(token or "", prompt, model=model_to_use, base_url=base_to_use, mock=mock_mode)
    if resp and not (isinstance(resp, str) and resp.startswith("__error__")):
        st.session_state['ai_summary'] = resp
        parsed = parse_ai_json(resp or "")
        if parsed:
            st.session_state['ai_summary_json'] = parsed
        st.session_state['last_results_ai_ts'] = st.session_state.get('last_results_ts')
        return resp
    return None


@st.cache_data(ttl=3600)
def fetch_tamu_models(api_key: str, base_url: str = "https://chat-api.tamu.ai", mock: bool = False):
    """Return a list of model ids from the TAMU /openai/models endpoint. Returns empty list on failure.
    Cached for 1 hour.
    """
    if mock:
        return [DEFAULT_TAMU_MODEL]
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
            # Resolve runtime values from session_state (avoid relying on outer names)
            token_to_use = st.session_state.get('tamu_api_key')
            model_to_use = str(st.session_state.get('tamu_model') or DEFAULT_TAMU_MODEL)
            base_to_use = str(st.session_state.get('tamu_base_url') or "https://chat-api.tamu.ai")
            mock_mode = bool(st.session_state.get('tamu_mock', False))
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
                resp = call_tamu_api(token_to_use or "", prompt, model=model_to_use, base_url=base_to_use, mock=mock_mode)
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

# Place lookup inputs: support pressing Enter by using on_change callback
def _lookup_place_action():
    place_name = st.session_state.get('place_input', '').strip()
    if not place_name:
        return
    with st.sidebar.spinner("Looking up place..."):
        geo = get_geocoding_results(place_name)
        if not geo:
            st.sidebar.error("No results found")
            return
        st.sidebar.success(f"Found: {geo.get('name')}, {geo.get('country')}")
        st.session_state['loc_lat'] = geo.get('latitude')
        st.session_state['loc_lon'] = geo.get('longitude')
        st.session_state['location_name'] = f"{geo.get('name')}, {geo.get('country')}"
        # Auto-fetch precipitation using stored place_start/place_end
        try:
            sd = st.session_state.get('place_start') or (datetime.today().date() - timedelta(days=365))
            ed = st.session_state.get('place_end') or datetime.today().date()
            df_precip = get_precipitation(geo.get('latitude'), geo.get('longitude'), sd, ed)
            if df_precip is not None and not df_precip.empty:
                st.session_state['rain_df'] = df_precip
                st.sidebar.success(f"Fetched {len(df_precip)} days for {geo.get('name')}")
                try:
                    mini_fig = px.bar(df_precip.tail(30), x='Date', y='Rain', labels={'Rain': 'mm'}, title='Last 30 days (preview)')
                    st.sidebar.plotly_chart(mini_fig, width=260)
                except Exception:
                    pass
            else:
                st.sidebar.info("Lookup succeeded but no precipitation data available for that period.")
        except Exception as e:
            st.sidebar.error(f"Auto-fetch failed: {e}")

# text_input will call _lookup_place_action on Enter (via on_change)
place = st.sidebar.text_input("Place name (e.g. Austin, TX)", value="", key='place_input', on_change=_lookup_place_action)
# Allow the user to pick start/end dates for place lookup (defaults to last 365 days)
place_start = st.sidebar.date_input("Place lookup start date", datetime.today().date() - timedelta(days=365), key='place_start')
place_end = st.sidebar.date_input("Place lookup end date", datetime.today().date(), key='place_end')
if st.sidebar.button("Lookup place"):
    # Trigger same action when pressing the button
    _lookup_place_action()

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

# Provide defaults so other code can reference these even when key not provided
# Ensure these are concrete types (avoid passing None to helper functions)
tamu_base_url = str(st.session_state.get('tamu_base_url') or "https://chat-api.tamu.ai")
tamu_mock = bool(st.session_state.get('tamu_mock', False))

# NOTE: summaries are now always generated automatically (background) after running a simulation.

# Convenience: quick reset of the TAMU model to the intended default if the user wants it
if st.sidebar.button("Reset TAMU model to default"):
    st.session_state['tamu_model'] = DEFAULT_TAMU_MODEL

# Only show detailed TAMU options when an API key is provided (cleaner sidebar)
if tamu_api_key:
    with st.sidebar.expander("TAMU options", expanded=False):
        tamu_base_url = st.text_input("TAMU API base URL", value=st.session_state.get('tamu_base_url', "https://chat-api.tamu.ai"))
        tamu_mock = st.checkbox("Mock TAMU AI (dev)", value=st.session_state.get('tamu_mock', False))
        # mirror into session_state so helper functions can read latest values
        st.session_state['tamu_api_key'] = tamu_api_key
        st.session_state['tamu_base_url'] = tamu_base_url
        st.session_state['tamu_mock'] = tamu_mock

        # Try to fetch available TAMU models and present as a dropdown when possible
        available_models = []
        if tamu_api_key and not tamu_mock:
            try:
                with st.spinner("Fetching TAMU models..."):
                    available_models = fetch_tamu_models(
                        tamu_api_key,
                        base_url=str(tamu_base_url or "https://chat-api.tamu.ai"),
                        mock=bool(tamu_mock),
                    )
            except Exception:
                available_models = []

        if available_models:
            # use session_state key so selection persists
            tamu_model = st.selectbox("TAMU Model", options=available_models, index=0, key='tamu_model')
        else:
            tamu_model = st.text_input("TAMU Model (enter id)", value=st.session_state.get('tamu_model', DEFAULT_TAMU_MODEL), key='tamu_model')

        # normalize effective model string for downstream calls
        effective_tamu_model = str(st.session_state.get('tamu_model') or tamu_model or DEFAULT_TAMU_MODEL)

        if st.button("Test TAMU API"):
            if not tamu_base_url:
                st.error("Set the TAMU API base URL first.")
            else:
                # DNS preflight
                try:
                    from urllib.parse import urlparse

                    parsed = urlparse(tamu_base_url)
                    host = parsed.hostname or tamu_base_url
                    try:
                        ip = socket.gethostbyname(host)
                        st.success(f"Resolved host {host} -> {ip}")
                    except Exception as dns_e:
                        st.warning(f"Could not resolve host {host}: {dns_e}")
                except Exception:
                    host = None

                try:
                    test_prompt = "Test connection from Hydrology app. Reply with short text 'OK'."
                    test_resp = call_tamu_api(
                        tamu_api_key or "",
                        test_prompt,
                        model=str(st.session_state.get('tamu_model') or effective_tamu_model or DEFAULT_TAMU_MODEL),
                        base_url=str(tamu_base_url or "https://chat-api.tamu.ai"),
                        mock=bool(tamu_mock),
                    )
                    if not test_resp:
                        st.error("No response from TAMU API. Check network and base URL.")
                    elif isinstance(test_resp, str) and test_resp.startswith("__error__"):
                        st.error("TAMU API call failed. See raw details below.")
                        with st.expander("TAMU raw error"):
                            st.code(test_resp)
                    else:
                        st.success("TAMU API reachable (response shown below).")
                        with st.expander("TAMU test response"):
                            st.code(test_resp)
                except Exception as e:
                    st.error(f"TAMU test failed: {e}")
else:
    # clear any prior tamu session keys when no api key present
    st.session_state.pop('tamu_api_key', None)
    st.session_state.pop('tamu_base_url', None)
    st.session_state.pop('tamu_mock', None)
    st.session_state.pop('tamu_model', None)


# parameter keys (used by auto-tune prompt)
param_keys = ['sh','sw','sstar','sfc','n','zr','ks','ew','emax','beta','s0']
# Build a parameter snapshot (used by prompts). Use stored session values or defaults if present.
p = {k: st.session_state.get(k, st.session_state.get('param_defaults', {}).get(k)) for k in param_keys}

# Auto-tune (TAMU) control placed BEFORE the parameter widgets so applying values updates widgets on this run
with st.sidebar.container():
    # Trigger an Auto-Tune request which stores parsed suggestions into session state.
    if st.button("Auto-Tune parameters", key="auto_tune"):
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

        resp = call_tamu_api(
            tamu_api_key or "",
            prompt,
            model=str(st.session_state.get('tamu_model') or DEFAULT_TAMU_MODEL),
            base_url=str(tamu_base_url or "https://chat-api.tamu.ai"),
            mock=bool(tamu_mock),
        )
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
                # Persist suggestions so they survive reruns (Apply is a separate click)
                st.session_state['tamu_suggestions'] = parsed

    # If we have persisted suggestions, show them (this makes Apply available across reruns)
    if st.session_state.get('tamu_suggestions'):
        st.sidebar.success("Received suggested parameters from TAMU")
        with st.sidebar.expander("Suggested parameters (TAMU)", expanded=True):
            suggestions = st.session_state.get('tamu_suggestions', {})
            st.json(suggestions)
            if st.button("Apply suggested parameters", key="apply_tamu_params"):
                # Save current params for undo
                hist = st.session_state.setdefault('param_history', [])
                hist.append({k: st.session_state.get(k) for k in param_keys})
                # Apply suggested values where present (read from persisted suggestions)
                for k, v in (suggestions or {}).items():
                    if k in param_keys:
                        try:
                            # allow numbers or numeric strings and clamp to allowed ranges
                            st.session_state[k] = clamp_param_value(k, v)
                        except Exception:
                            # if conversion fails, still set the raw value
                            st.session_state[k] = v
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

# --- SIDEBAR: PARAMETERS ---

# Use session_state-backed widgets for parameters so they can be updated programmatically
st.sidebar.subheader("Model Parameters")
st.sidebar.number_input("Hygroscopic Point (sh)", 0.0, 1.0, value=st.session_state.get('sh', 0.08), format="%.2f", key='sh')
st.sidebar.number_input("Wilting Point (sw)", 0.0, 1.0, value=st.session_state.get('sw', 0.11), format="%.2f", key='sw')
st.sidebar.number_input("Stomatal Closure (s*)", 0.0, 1.0, value=st.session_state.get('sstar', 0.33), format="%.2f", key='sstar')
st.sidebar.number_input("Field Capacity (sfc)", 0.0, 1.0, value=st.session_state.get('sfc', 0.40), format="%.2f", key='sfc')
st.sidebar.number_input("Porosity (n)", 0.1, 1.0, value=st.session_state.get('n', 0.55), format="%.2f", key='n')
st.sidebar.number_input("Root Depth (Zr) [mm]", 10.0, 2000.0, value=st.session_state.get('zr', 500.0), key='zr')
st.sidebar.number_input("Saturated Conductivity (Ks) [mm/day]", 0.0, 1000.0, value=st.session_state.get('ks', 200.0), key='ks')
st.sidebar.number_input("Evap at Wilting (Ew) [mm/day]", 0.0, 10.0, value=st.session_state.get('ew', 0.1), key='ew')
st.sidebar.number_input("Max Evap (Emax) [mm/day]", 0.0, 20.0, value=st.session_state.get('emax', 5.0), key='emax')
st.sidebar.number_input("Leakage Parameter (Beta)", 0.0, 20.0, value=st.session_state.get('beta', 3.0), key='beta')
st.sidebar.slider("Initial Soil Moisture (s0)", 0.0, 1.0, value=st.session_state.get('s0', 0.3), key='s0')

# Build parameter dict from session_state
p = {k: st.session_state.get(k) for k in param_keys}
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
                # mark that we've rendered this results set on this run so persisted rendering doesn't duplicate
                st.session_state['last_results_rendered_ts'] = st.session_state['last_results_ts']
            except Exception:
                # if session_state can't store the DataFrame for any reason, ignore
                pass

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

            # Summary placeholder (renders right after metrics). We render existing summary
            # if available, otherwise provide a Generate button that uses current session settings.
            summary_placeholder = st.empty()
            try:
                def _render_summary_in_placeholder(resp_text):
                    parsed = parse_ai_json(resp_text or "")
                    with summary_placeholder.container():
                        st.markdown("### AI Summary")
                        plain = resp_text.split('JSON_START')[0].strip() if resp_text and 'JSON_START' in resp_text else (resp_text or "")
                        if plain:
                            st.info(plain)
                        if parsed:
                            st.subheader("AI structured summary")
                            st.json(parsed)
                        else:
                            st.write("No structured JSON available in AI response.")

                # If an AI summary has already been generated for these results, show it
                if st.session_state.get('ai_summary') and st.session_state.get('last_results_ai_ts') == st.session_state.get('last_results_ts'):
                    _render_summary_in_placeholder(st.session_state.get('ai_summary'))
                else:
                    # show a generate button in the placeholder
                    with summary_placeholder.container():
                        st.markdown("### AI Summary")
                        st.write("No AI summary generated for these results yet.")
                        if st.button("Generate AI Summary (include model results)", key="generate_ai_after_run"):
                            # Prefer persisted results so this button works across reruns
                            rs = st.session_state.get('last_results') or results
                            if rs is None:
                                st.error("No results available to summarize. Run a simulation first.")
                            else:
                                resp = generate_and_store_summary(rs)
                                if resp:
                                    _render_summary_in_placeholder(resp)
                                else:
                                    st.error("AI summary generation failed. See raw response (if any) in session state.")

                # If no AI summary exists for these results yet, generate one synchronously (legacy behavior).
                try:
                    need_gen = bool(st.session_state.get('last_results_ts') and st.session_state.get('last_results_ai_ts') != st.session_state.get('last_results_ts'))
                    if need_gen:
                        rs = st.session_state.get('last_results') or results
                        if rs is not None:
                            resp = generate_and_store_summary(rs)
                            if resp:
                                _render_summary_in_placeholder(resp)
                            else:
                                # If generation failed but we have some stored ai_summary for this ts, show it
                                if st.session_state.get('ai_summary') and st.session_state.get('last_results_ai_ts') == st.session_state.get('last_results_ts'):
                                    _render_summary_in_placeholder(st.session_state.get('ai_summary'))
                                else:
                                    st.warning("AI summary generation did not return a result. Check TAMU API settings.")
                except Exception:
                    pass
            except Exception:
                pass
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Check your CSV columns. They must be 'Date' and 'Rain'.")

# If there are persisted results from a previous run, render them so summaries persist across reruns
# Avoid duplicating the immediate results we just rendered in the Run Simulation flow by
# checking a rendered timestamp. If the rendered timestamp matches the results timestamp,
# we've already shown them above and can skip the persisted block.
if 'last_results' in st.session_state and st.session_state.get('last_results_rendered_ts') != st.session_state.get('last_results_ts'):
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