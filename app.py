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
    # HTTP header values must be Latin-1 (per http.client). Some copy/pasted API keys
    # or base URLs can include non-Latin characters (e.g. an em-dash) which cause
    # a UnicodeEncodeError when requests tries to send headers. Sanitize header
    # values by dropping characters that can't be encoded as latin-1 so the
    # request fails more clearly on auth rather than raising during header encoding.
    def _sanitize_header_value(v):
        if v is None:
            return ''
        try:
            # Coerce to str, then encode->decode latin-1 ignoring unencodable chars
            return str(v).encode('latin-1', errors='ignore').decode('latin-1')
        except Exception:
            return str(v)

    safe_key = _sanitize_header_value(api_key)
    headers = {"Authorization": f"Bearer {safe_key}", "Content-Type": "application/json"}
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


def call_openai_api(api_key: str, prompt: str, model: str = "gpt-4", timeout: int = 30):
    """Call the OpenAI chat completions endpoint and return assistant text or an error string prefixed with __error__.
    Uses POST https://api.openai.com/v1/chat/completions
    """
    if not api_key:
        return "__error__:NoAPIKey:No OpenAI API key provided"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
    }
    try:
        r = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            return f"__error__:HTTP_{r.status_code}:{r.text}"
        data = r.json()
        choices = data.get("choices") or []
        if choices:
            message = choices[0].get("message", {})
            content = message.get("content")
            if content:
                return content
        return "__error__:NoContent:No content in response"
    except requests.exceptions.RequestException as e:
        return f"__error__:{e.__class__.__name__}:{str(e)}"
    except Exception as e:
        return f"__error__:UnexpectedError:{str(e)}"


def call_gemini_api(api_key: str, prompt: str, model: str = "gemini-1.5-flash", timeout: int = 30):
    """Call the Google Gemini API and return response text or an error string prefixed with __error__.
    Uses REST API endpoint: https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent
    """
    if not api_key:
        return "__error__:NoAPIKey:No Gemini API key provided"

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
    }
    try:
        # API key is passed as query parameter for Gemini
        r = requests.post(f"{url}?key={api_key}", headers=headers, json=payload, timeout=timeout)
        if r.status_code != 200:
            return f"__error__:HTTP_{r.status_code}:{r.text}"
        data = r.json()
        candidates = data.get("candidates") or []
        if candidates:
            content = candidates[0].get("content", {})
            parts = content.get("parts") or []
            if parts:
                text = parts[0].get("text")
                if text:
                    return text
        return "__error__:NoContent:No content in response"
    except requests.exceptions.RequestException as e:
        return f"__error__:{e.__class__.__name__}:{str(e)}"
    except Exception as e:
        return f"__error__:UnexpectedError:{str(e)}"


def call_ai_api(prompt: str, api_provider: str = "tamu", api_key: str = "", model: str = "", base_url: str = "https://chat-api.tamu.ai", mock: bool = False, timeout: int = 30):
    """Unified function to call various AI APIs based on selected provider.
    Returns assistant text or an error string prefixed with __error__.
    """
    if api_provider.lower() == "openai":
        return call_openai_api(api_key, prompt, model or "gpt-4", timeout)
    elif api_provider.lower() == "gemini":
        return call_gemini_api(api_key, prompt, model or "gemini-1.5-flash", timeout)
    else:  # default to TAMU
        return call_tamu_api(api_key, prompt, model or DEFAULT_TAMU_MODEL, base_url, mock, timeout)


def generate_and_store_summary(results_df):
    """Generate AI summary for a given results DataFrame and store into session_state.
    Uses session_state values for token/model/base_url/mock.
    On success, stores 'ai_summary' and 'ai_summary_json'.
    On failure, stores an error message in 'ai_summary_error'.
    """
    # Clear previous summary/error states before generating a new one
    st.session_state.pop('ai_summary', None)
    st.session_state.pop('ai_summary_json', None)
    st.session_state.pop('ai_summary_error', None)

    if results_df is None:
        return None
    
    # Get the selected AI provider and corresponding API key
    provider = st.session_state.get('ai_provider', 'tamu')
    if provider == 'tamu':
        token = st.session_state.get('tamu_api_key')
        model_to_use = str(st.session_state.get('tamu_model') or DEFAULT_TAMU_MODEL)
        base_url = str(st.session_state.get('tamu_base_url') or "https://chat-api.tamu.ai")
        mock_mode = bool(st.session_state.get('tamu_mock', False))
    elif provider == 'openai':
        token = st.session_state.get('openai_api_key')
        model_to_use = st.session_state.get('openai_model', 'gpt-4')
        base_url = ""
        mock_mode = False
    elif provider == 'gemini':
        token = st.session_state.get('gemini_api_key')
        model_to_use = st.session_state.get('gemini_model', 'gemini-1.5-flash')
        base_url = ""
        mock_mode = False
    else:
        st.session_state['ai_summary_error'] = "No AI provider selected"
        return None
    
    try:
        tr = float(results_df['Rain'].sum())
        te = float(results_df['et'].sum())
        tq = float(results_df['q'].sum())
        ms = float(results_df['s'].mean())
    except Exception:
        tr = te = tq = ms = 0.0

    # 1. Get Context: Retrieve soil parameters to make the AI smarter
    sw_val = st.session_state.get('sw', 0.11)
    sfc_val = st.session_state.get('sfc', 0.40)
    sstar_val = st.session_state.get('sstar', 0.33)
    sh_val = st.session_state.get('sh', 0.08)
    n_val = st.session_state.get('n', 0.55)
    zr_val = st.session_state.get('zr', 500.0)
    ks_val = st.session_state.get('ks', 200.0)

    prompt = f"""
Act as an expert Hydrologist and Agricultural Consultant. Analyze the water balance simulation for {st.session_state.get('location_name','(unknown)')} (Lat: {st.session_state.get('loc_lat', 'unknown')}, Lon: {st.session_state.get('loc_lon', 'unknown')}).

Physical Context (Critical for analysis):
- Soil Wilting Point (sw): {sw_val} (Below this, plants suffer water stress).
- Field Capacity (sfc): {sfc_val} (Above this, water is lost to drainage/runoff).
- Stomatal Closure Point (s*): {sstar_val} (Plants start closing stomata below this).
- Hygroscopic Point (sh): {sh_val} (Water unavailable to plants).
- Porosity (n): {n_val} (Total void space).
- Root Depth (Zr): {zr_val} mm (Active soil layer depth).
- Saturated Conductivity (Ks): {ks_val} mm/day (Max drainage rate).

Simulation Results for {st.session_state.get('location_name','(unknown)')}:
- Total Rain: {tr:.2f} mm
- Total ET (Productive Use): {te:.2f} mm
- Total Runoff (Loss): {tq:.2f} mm
- Average Soil Moisture: {ms:.3f}

Task:
1. Write a 3-5 sentence technical summary specific to {st.session_state.get('location_name','(unknown)')}. Compare the 'Average Soil Moisture' to the Wilting Point and Field Capacity thresholds. Was the soil healthy, too dry (stressed), or saturated (leaking)? Use markdown bolding (e.g. **value**) for all numeric values and statistics in the text.
2. Evaluate the "Water Efficiency" at this location: Did most rain become productive ET, or was it lost to runoff? Consider typical climate patterns for this region.
3. Explain the soil conditions and what they indicate about water availability at {st.session_state.get('location_name','(unknown)')}.
4. Return a JSON object exactly between markers JSON_START and JSON_END with keys: 
   - 'recommendation': Specific, location-appropriate adjustment (e.g., "For {st.session_state.get('location_name','(unknown)')}, implement drainage" or "Apply irrigation to prevent hitting wilting point")
   - 'simple_summary': Very simple, non-technical explanation for this location
   - 'soil_conditions_reasoning': Explanation of what the measured soil moisture and parameters reveal about {st.session_state.get('location_name','(unknown)')}'s soil behavior
   - 'location_specific_factors': How regional climate, soil type, and vegetation at {st.session_state.get('location_name','(unknown)')} influence these results
5. All recommendations and reasoning should explicitly reference {st.session_state.get('location_name','(unknown)')}.
"""
    resp = call_ai_api(
        prompt,
        api_provider=provider,
        api_key=token or "",
        model=model_to_use,
        base_url=base_url,
        mock=mock_mode
    )

    if resp and not (isinstance(resp, str) and resp.startswith("__error__")):
        st.session_state['ai_summary'] = resp
        parsed = parse_ai_json(resp or "")
        if parsed:
            st.session_state['ai_summary_json'] = parsed
        else:
            st.session_state['ai_summary_error'] = "AI response received, but failed to parse structured JSON."
        st.session_state['last_results_ai_ts'] = st.session_state.get('last_results_ts')
        return resp
    elif isinstance(resp, str) and resp.startswith("__error__"):
        st.session_state['ai_summary_error'] = f"AI request failed: {resp}"
    else:
        st.session_state['ai_summary_error'] = f"No response from {provider.upper()} API. Check API key and network connectivity."
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


def fetch_soilgrids_data(lat: float, lon: float):
    """Fetch soil properties from SoilGrids REST API for given coordinates.
    Returns a dict with soil properties or None on failure.
    https://rest.isric.org/soilgrids/v2.0/docs
    """
    try:
        url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
        params = {
            "lon": float(lon),
            "lat": float(lat),
            "property": ["clay", "sand", "silt", "organic_carbon"],
            "depth": ["0-5cm", "5-15cm", "15-30cm", "30-60cm"],
        }
        r = requests.get(url, params=params, timeout=15)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


def infer_parameters_from_soilgrids(soilgrids_data: dict):
    """Infer water balance model parameters from SoilGrids data.
    Uses soil texture (clay/sand/silt percentages) to estimate parameters.
    Returns a dict with estimated parameters and soil_type description.
    """
    if not soilgrids_data or 'properties' not in soilgrids_data:
        return None
    
    try:
        props = soilgrids_data.get('properties', {})
        layers = props.get('layers', [])
        if not layers:
            return None
        
        # Use 0-5cm layer for topsoil properties
        layer = layers[0]
        depths = layer.get('depths', [])
        if not depths:
            return None
        
        depth_0_5 = depths[0]
        
        # Extract soil texture percentages (0-5cm layer, mean values)
        clay_mean = depth_0_5.get('clay', {}).get('mean', 20.0) / 1000.0  # Convert from g/kg to fraction
        sand_mean = depth_0_5.get('sand', {}).get('mean', 50.0) / 1000.0
        silt_mean = depth_0_5.get('silt', {}).get('mean', 30.0) / 1000.0
        org_carbon = depth_0_5.get('organic_carbon', {}).get('mean', 15.0) / 1000.0  # Convert from dg/kg to fraction
        
        # Normalize to fractions (0-1)
        total = clay_mean + sand_mean + silt_mean
        if total > 0:
            clay = clay_mean / total
            sand = sand_mean / total
            silt = silt_mean / total
        else:
            clay, sand, silt = 0.2, 0.5, 0.3
        
        # Classify soil texture (USDA soil texture triangle)
        soil_type = classify_soil_texture(clay, sand, silt)
        
        # Estimate water balance parameters based on soil texture
        # Using typical values from soil physics literature
        params = estimate_water_balance_parameters(clay, sand, silt, org_carbon)
        params['soil_type'] = soil_type
        params['clay_fraction'] = clay
        params['sand_fraction'] = sand
        params['silt_fraction'] = silt
        params['organic_carbon'] = org_carbon
        
        return params
    except Exception:
        return None


def classify_soil_texture(clay: float, sand: float, silt: float) -> str:
    """Classify soil texture based on USDA soil texture triangle.
    Returns soil texture class name.
    """
    clay_pct = clay * 100
    sand_pct = sand * 100
    silt_pct = silt * 100
    
    # USDA soil texture classification rules
    if clay_pct < 27 and sand_pct > 52 and silt_pct < 50:
        if sand_pct > 86:
            return "Sand"
        return "Sandy Loam"
    elif clay_pct < 27 and (sand_pct <= 52 or silt_pct >= 50):
        if silt_pct >= 80:
            return "Silt"
        elif silt_pct >= 50:
            return "Silt Loam"
        return "Loam"
    elif clay_pct >= 27 and clay_pct < 40:
        if sand_pct > 52:
            return "Sandy Clay Loam"
        return "Clay Loam"
    elif clay_pct >= 40:
        if sand_pct > 52:
            return "Sandy Clay"
        elif silt_pct >= 60:
            return "Silty Clay"
        return "Clay"
    return "Loam"


def estimate_water_balance_parameters(clay: float, sand: float, silt: float, org_carbon: float) -> dict:
    """Estimate Laio et al. water balance parameters from soil texture and organic matter.
    Uses empirical pedotransfer functions.
    """
    clay_pct = clay * 100
    sand_pct = sand * 100
    
    # Adjust porosity (n) based on clay and organic content
    # Sandy soils: ~0.4, loamy: ~0.45-0.50, clayey: ~0.50-0.55
    base_porosity = 0.35 + (clay_pct * 0.002) + (org_carbon * 20)
    n = min(max(base_porosity, 0.35), 0.60)
    
    # Field capacity (sfc) - higher in finer-textured soils
    sfc = 0.15 + (clay_pct * 0.003) + (org_carbon * 10)
    sfc = min(max(sfc, 0.15), 0.50)
    
    # Wilting point (sw) - also increases with clay
    sw = max(0.05, sfc * 0.4)
    
    # Stomatal closure point (s*) - typically 0.5-0.7 of field capacity
    sstar = sfc * 0.65
    
    # Hygroscopic point (sh) - typically 0.05-0.15
    sh = max(0.05, sw * 0.5)
    
    # Saturated conductivity (Ks) - decreases with clay content
    # Sandy: 300-500 mm/day, loamy: 100-300, clayey: 10-100
    ks = max(10, 500 * (1 - (clay_pct / 100) ** 2))
    
    # Root depth (Zr) - typical values 300-700mm, slightly higher in sandy soils
    zr = 400 + (sand_pct * 3)
    zr = min(max(zr, 300), 750)
    
    # ET parameters - slightly higher in finer-textured soils
    emax = 5.0 + (clay_pct * 0.05)
    emax = min(max(emax, 3.0), 8.0)
    
    ew = emax * 0.02
    
    # Leakage parameter (beta) - lower in sandy soils, higher in clayey
    beta = 0.5 + (clay_pct * 0.05)
    beta = min(max(beta, 0.2), 5.0)
    
    # Initial soil moisture - set to halfway between sh and sfc
    s0 = (sh + sfc) / 2
    
    return {
        'sh': round(sh, 3),
        'sw': round(sw, 3),
        'sstar': round(sstar, 3),
        'sfc': round(sfc, 3),
        'n': round(n, 3),
        'zr': round(zr, 1),
        'ks': round(ks, 1),
        'ew': round(ew, 3),
        'emax': round(emax, 2),
        'beta': round(beta, 2),
        's0': round(s0, 3),
    }


def show_ai_summary_block(results, button_key: str = "generate_ai_summary_persistent"):
    """Render a formatted AI summary and provide controls to generate it."""
    if results is None:
        return

    st.markdown("---")
    st.subheader("AI-Generated Summary")

    # Display any error from the last generation attempt
    if 'ai_summary_error' in st.session_state:
        st.error(st.session_state.get('ai_summary_error'))

    # Display existing summary in a clean, formatted way
    if 'ai_summary' in st.session_state:
        raw_summary = st.session_state.get('ai_summary', '')
        parsed_json = st.session_state.get('ai_summary_json')

        # 1. Parse narrative text from raw response
        narrative = raw_summary
        if "JSON_START" in narrative:
            narrative = narrative.split("JSON_START")[0].strip()

        # 2. Display Narrative
        if narrative:
            st.markdown(narrative)
        # Fallback for malformed response with no narrative part
        elif not parsed_json:
            st.markdown(raw_summary)

        # 3. Display Quick Summary and Recommendation First (at top)
        if parsed_json and isinstance(parsed_json, dict):
            simple_summary = parsed_json.get('simple_summary')
            if simple_summary:
                st.info(f"**Simple Summary:** {simple_summary}", icon="👋")

            recommendation = parsed_json.get('recommendation')
            if recommendation:
                st.success(f"**Insight:** {recommendation}", icon="💡")
            
            st.markdown("---")

            soil_reasoning = parsed_json.get('soil_conditions_reasoning')
            if soil_reasoning:
                st.markdown(f"**Soil Conditions Analysis:** {soil_reasoning}")

            location_factors = parsed_json.get('location_specific_factors')
            if location_factors:
                st.markdown(f"**Location-Specific Factors:** {location_factors}")
    else:
        # Only show this if there's no error message either
        if 'ai_summary_error' not in st.session_state:
            st.info("Click 'Generate / Regenerate AI Summary' below to get an analysis of the results (Requires TAMU API key).")

    # Controls to generate/regenerate the summary
    with st.expander("Generate / Regenerate AI Summary", expanded=('ai_summary' not in st.session_state)):

        button_text = "Regenerate AI Summary" if 'ai_summary' in st.session_state else "Generate AI Summary"

        if st.button(button_text, key=button_key):
            with st.spinner("Contacting TAMU AI for a summary..."):
                generate_and_store_summary(results)
                st.rerun()


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
url = "https://doi.org/10.1016/S0309-1708(01)00005-7"
st.markdown("This app simulates soil moisture dynamics based on the [**Laio et al. (2001)** model.](%s)" % url)

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

# --- AI API Settings (supports TAMU, OpenAI, and Gemini) ---
st.sidebar.header("AI API Settings")
st.sidebar.markdown("Select which AI service to use for auto-tune and summaries")

# Choose AI provider
ai_provider = st.sidebar.radio(
    "Select AI Provider:",
    options=["TAMU AI", "OpenAI (ChatGPT)", "Google Gemini"],
    index=0,
    help="Choose which API to use. You only need to enter a key for your chosen provider."
)

# Map provider name to internal identifier
provider_map = {
    "TAMU AI": "tamu",
    "OpenAI (ChatGPT)": "openai",
    "Google Gemini": "gemini"
}
selected_provider = provider_map[ai_provider]
st.session_state['ai_provider'] = selected_provider

# API Key input (provider-specific)
if ai_provider == "TAMU AI":
    api_key_input = st.sidebar.text_input("TAMU AI API Key (optional)", type="password")
    st.session_state['tamu_api_key'] = api_key_input
    st.session_state['openai_api_key'] = None
    st.session_state['gemini_api_key'] = None
    
    tamu_base_url = str(st.session_state.get('tamu_base_url') or "https://chat-api.tamu.ai")
    tamu_mock = bool(st.session_state.get('tamu_mock', False))
    st.session_state.setdefault('tamu_base_url', tamu_base_url)
    st.session_state.setdefault('tamu_mock', tamu_mock)
    
    if st.sidebar.button("Reset TAMU model to default"):
        st.session_state['tamu_model'] = DEFAULT_TAMU_MODEL
    
    if api_key_input:
        with st.sidebar.expander("TAMU options", expanded=False):
            tamu_base_url = st.text_input("TAMU API base URL", value=st.session_state.get('tamu_base_url', "https://chat-api.tamu.ai"))
            tamu_mock = st.checkbox("Mock TAMU AI (dev)", value=st.session_state.get('tamu_mock', False))
            st.session_state['tamu_base_url'] = tamu_base_url
            st.session_state['tamu_mock'] = tamu_mock
            
            # Try to fetch available TAMU models
            available_models = []
            if api_key_input and not tamu_mock:
                try:
                    with st.spinner("Fetching TAMU models..."):
                        available_models = fetch_tamu_models(
                            api_key_input,
                            base_url=str(tamu_base_url or "https://chat-api.tamu.ai"),
                            mock=bool(tamu_mock),
                        )
                except Exception:
                    available_models = []
            
            if available_models:
                tamu_model = st.selectbox("TAMU Model", options=available_models, index=0, key='tamu_model')
            else:
                tamu_model = st.text_input("TAMU Model (enter id)", value=st.session_state.get('tamu_model', DEFAULT_TAMU_MODEL), key='tamu_model')

elif ai_provider == "OpenAI (ChatGPT)":
    api_key_input = st.sidebar.text_input("OpenAI API Key", type="password", help="Get your key from https://platform.openai.com/api-keys")
    st.session_state['openai_api_key'] = api_key_input
    st.session_state['tamu_api_key'] = None
    st.session_state['gemini_api_key'] = None
    
    if api_key_input:
        with st.sidebar.expander("OpenAI options", expanded=False):
            openai_model = st.selectbox(
                "OpenAI Model",
                options=["gpt-4", "gpt-4-turbo", "gpt-3.5-turbo"],
                key='openai_model'
            )
            st.session_state['openai_model'] = openai_model

elif ai_provider == "Google Gemini":
    api_key_input = st.sidebar.text_input("Gemini API Key", type="password", help="Get your key from https://aistudio.google.com/apikey")
    st.session_state['gemini_api_key'] = api_key_input
    st.session_state['tamu_api_key'] = None
    st.session_state['openai_api_key'] = None
    
    if api_key_input:
        with st.sidebar.expander("Gemini options", expanded=False):
            gemini_model = st.selectbox(
                "Gemini Model",
                options=["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro"],
                key='gemini_model'
            )
            st.session_state['gemini_model'] = gemini_model

# Test API connection if key is provided
if api_key_input:
    if st.sidebar.button("Test API Connection"):
        with st.spinner("Testing API connection..."):
            test_prompt = "Test connection. Reply with just 'OK'."
            
            if selected_provider == "tamu":
                test_resp = call_tamu_api(
                    api_key_input,
                    test_prompt,
                    model=str(st.session_state.get('tamu_model') or DEFAULT_TAMU_MODEL),
                    base_url=str(st.session_state.get('tamu_base_url') or "https://chat-api.tamu.ai"),
                    mock=bool(st.session_state.get('tamu_mock', False))
                )
            elif selected_provider == "openai":
                test_resp = call_openai_api(
                    api_key_input,
                    test_prompt,
                    model=st.session_state.get('openai_model', 'gpt-4')
                )
            else:  # gemini
                test_resp = call_gemini_api(
                    api_key_input,
                    test_prompt,
                    model=st.session_state.get('gemini_model', 'gemini-1.5-flash')
                )
            
            if isinstance(test_resp, str) and test_resp.startswith("__error__"):
                st.error(f"API test failed: {test_resp}")
            else:
                st.success("API connection successful!")
                with st.expander("Test response"):
                    st.write(test_resp)


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

        loc_name = st.session_state.get('location_name', 'Unknown Location')
        loc_lat = st.session_state.get('loc_lat', 'Unknown')
        loc_lon = st.session_state.get('loc_lon', 'Unknown')

        prompt = f"""
Act as an expert Hydrologist. Suggest calibrated model parameters for the Laio et al. (2001) water-balance model appropriate for the following location and climate.

Location: {loc_name} (Lat: {loc_lat}, Lon: {loc_lon})

Current Simulation Context ({sample_days} days):
- Total Rain: {lr:.2f} mm
- Simulated ET: {let:.2f} mm
- Simulated Runoff: {lq:.2f} mm
- Mean Soil Moisture: {ls:.3f}

Current Parameters:
{json.dumps(p)}

Task:
Suggest a tuned set of parameters ('sh', 'sw', 'sstar', 'sfc', 'n', 'zr', 'ks', 'ew', 'emax', 'beta', 's0') that are physically realistic for the soil texture and vegetation type typically found at {loc_name}. Adjust values to ensure water balance components (ET vs Runoff) are reasonable for this climate.

Return a JSON object exactly between markers JSON_START and JSON_END with the following structure:
- Include all parameter values: sh, sw, sstar, sfc, n, zr, ks, ew, emax, beta, s0
- Include 'soil_type': Inferred dominant soil texture class (e.g., "Sandy Loam", "Clay Loam")
- Include 'soil_reasoning': Brief explanation of why you chose that soil type and parameter values based on {loc_name}'s climate and typical pedology
- Include 'vegetation_type': Expected dominant vegetation for this location
- Include 'climate_factors': How the local climate (precipitation patterns, temperature seasonality) influences the calibration
"""

        # Get the selected AI provider and API key
        provider = st.session_state.get('ai_provider', 'tamu')
        if provider == 'tamu':
            api_key = st.session_state.get('tamu_api_key')
            model = st.session_state.get('tamu_model', DEFAULT_TAMU_MODEL)
            base_url = st.session_state.get('tamu_base_url', 'https://chat-api.tamu.ai')
            mock = st.session_state.get('tamu_mock', False)
        elif provider == 'openai':
            api_key = st.session_state.get('openai_api_key')
            model = st.session_state.get('openai_model', 'gpt-4')
            base_url = ""
            mock = False
        elif provider == 'gemini':
            api_key = st.session_state.get('gemini_api_key')
            model = st.session_state.get('gemini_model', 'gemini-1.5-flash')
            base_url = ""
            mock = False
        else:
            st.error("No AI provider selected. Please configure an API provider first.")
            api_key = None

        if not api_key:
            st.error(f"No API key found for {provider.upper()}. Enter your API key in the 'AI API Settings' section.")
        else:
            resp = call_ai_api(
                prompt,
                api_provider=provider,
                api_key=api_key,
                model=model,
                base_url=base_url,
                mock=mock
            )
            if not resp:
                st.error(f"No response from {provider.upper()} API. Check your API key and network.")
            elif isinstance(resp, str) and resp.startswith("__error__"):
                st.error(f"{provider.upper()} API call failed. See raw details below.")
                with st.expander(f"{provider.upper()} raw error"):
                    st.code(resp)
            else:
                parsed = parse_ai_json(resp or "")
                if not parsed or not isinstance(parsed, dict):
                    st.error(f"{provider.upper()} did not return parsable JSON. See raw response.")
                    with st.expander(f"{provider.upper()} raw response"):
                        st.code(resp)
                else:
                    # Persist suggestions so they survive reruns (Apply is a separate click)
                    st.session_state['ai_suggestions'] = parsed
                    st.sidebar.success(f"Received suggested parameters from {provider.upper()}")

    # If we have persisted suggestions, show them (this makes Apply available across reruns)
    if st.session_state.get('ai_suggestions'):
        provider = st.session_state.get('ai_provider', 'tamu').upper()
        with st.sidebar.expander(f"Suggested parameters ({provider})", expanded=True):
            suggestions = st.session_state.get('ai_suggestions', {})

            
            # Display reasoning if available
            if suggestions.get('soil_type'):
                st.markdown(f"**Inferred Soil Type:** {suggestions['soil_type']}")
            if suggestions.get('vegetation_type'):
                st.markdown(f"**Expected Vegetation:** {suggestions['vegetation_type']}")
            if suggestions.get('soil_reasoning'):
                st.info(f"**Reasoning:** {suggestions['soil_reasoning']}")
            if suggestions.get('climate_factors'):
                st.markdown(f"**Climate Factors:** {suggestions['climate_factors']}")
            
            st.markdown("---")
            st.markdown("**Parameter Values:**")
            # Display only the parameter values
            param_display = {k: v for k, v in suggestions.items() if k in param_keys}
            st.json(param_display)
            
            if st.button("Apply suggested parameters", key="apply_ai_suggestions"):
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
                st.rerun()
            if st.button("Undo last parameter apply", key="undo_ai_suggestions"):
                hist = st.session_state.get('param_history', [])
                if hist:
                    last_params = hist.pop()
                    for k, v in last_params.items():
                        st.session_state[k] = v
                    st.success("Reverted to previous parameters")
                    st.rerun()
                else:
                    st.info("No previous parameter snapshot available to undo.")

    # SoilGrids parameter estimation based on coordinates
    st.markdown("---")
    st.sidebar.subheader("Get Soil Parameters from SoilGrids")
    st.sidebar.markdown("*Fetch soil data based on your location coordinates*")
    
    if st.sidebar.button("Fetch soil parameters from SoilGrids", key="fetch_soilgrids"):
        loc_lat = st.session_state.get('loc_lat')
        loc_lon = st.session_state.get('loc_lon')
        loc_name = st.session_state.get('location_name', 'Unknown Location')
        
        if loc_lat is None or loc_lon is None:
            st.sidebar.error("Please select a location first (use place lookup or coordinates).")
        else:
            with st.sidebar.spinner("Fetching soil data from SoilGrids..."):
                soilgrids_data = fetch_soilgrids_data(loc_lat, loc_lon)
                if soilgrids_data is None:
                    st.sidebar.error("Failed to fetch soil data from SoilGrids. Check your coordinates.")
                else:
                    params = infer_parameters_from_soilgrids(soilgrids_data)
                    if params is None:
                        st.sidebar.error("Could not parse SoilGrids data.")
                    else:
                        st.sidebar.success("Successfully fetched soil parameters from SoilGrids!")
                        st.session_state['soilgrids_params'] = params

    # Display SoilGrids results if available
    if st.session_state.get('soilgrids_params'):
        st.sidebar.success("Soil parameters from SoilGrids")
        with st.sidebar.expander("SoilGrids Results", expanded=True):
            params = st.session_state.get('soilgrids_params', {})
            
            # Display soil classification and properties
            if params.get('soil_type'):
                st.markdown(f"**Soil Type:** {params['soil_type']}")
            if params.get('clay_fraction') is not None:
                clay_pct = round(params['clay_fraction'] * 100, 1)
                sand_pct = round(params['sand_fraction'] * 100, 1)
                silt_pct = round(params['silt_fraction'] * 100, 1)
                st.markdown(f"**Texture:** Clay {clay_pct}% · Sand {sand_pct}% · Silt {silt_pct}%")
            if params.get('organic_carbon') is not None:
                org_pct = round(params['organic_carbon'] * 100, 2)
                st.markdown(f"**Organic Carbon:** {org_pct}%")
            
            st.markdown("---")
            st.markdown("**Estimated Parameters:**")
            # Display parameter values
            param_display = {k: v for k, v in params.items() if k in param_keys}
            st.json(param_display)
            
            if st.sidebar.button("Apply SoilGrids parameters", key="apply_soilgrids_params"):
                # Save current params for undo
                hist = st.session_state.setdefault('param_history', [])
                hist.append({k: st.session_state.get(k) for k in param_keys})
                # Apply SoilGrids parameters
                for k, v in params.items():
                    if k in param_keys:
                        try:
                            st.session_state[k] = clamp_param_value(k, v)
                        except Exception:
                            st.session_state[k] = v
                st.sidebar.success("Applied SoilGrids parameters")
                st.rerun()

# --- SIDEBAR: PARAMETERS ---

# Initialize session state defaults for all parameters if not already set
param_defaults = {
    'sh': 0.08,
    'sw': 0.11,
    'sstar': 0.33,
    'sfc': 0.40,
    'n': 0.55,
    'zr': 500.0,
    'ks': 200.0,
    'ew': 0.1,
    'emax': 5.0,
    'beta': 3.0,
    's0': 0.3,
}
for key, default_val in param_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# Use session_state-backed widgets for parameters so they can be updated programmatically
st.sidebar.subheader("Model Parameters")
st.sidebar.number_input("Hygroscopic Point (sh)", 0.0, 1.0, format="%.2f", key='sh')
st.sidebar.number_input("Wilting Point (sw)", 0.0, 1.0, format="%.2f", key='sw')
st.sidebar.number_input("Stomatal Closure (s*)", 0.0, 1.0, format="%.2f", key='sstar')
st.sidebar.number_input("Field Capacity (sfc)", 0.0, 1.0, format="%.2f", key='sfc')
st.sidebar.number_input("Porosity (n)", 0.1, 1.0, format="%.2f", key='n')
st.sidebar.number_input("Root Depth (Zr) [mm]", 10.0, 2000.0, key='zr')
st.sidebar.number_input("Saturated Conductivity (Ks) [mm/day]", 0.0, 1000.0, key='ks')
st.sidebar.number_input("Evap at Wilting (Ew) [mm/day]", 0.0, 10.0, key='ew')
st.sidebar.number_input("Max Evap (Emax) [mm/day]", 0.0, 20.0, key='emax')
st.sidebar.number_input("Leakage Parameter (Beta)", 0.0, 20.0, key='beta')
st.sidebar.slider("Initial Soil Moisture (s0)", 0.0, 1.0, key='s0')

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
        try:
            # Ensure 'Date' is datetime
            if 'Date' in rain_df.columns:
                rain_df['Date'] = pd.to_datetime(rain_df['Date'])
            
            # 1. RUN MODEL
            results = water_balance_simulation(p, rain_df)
            
            # 2. SAVE RESULTS TO SESSION STATE
            st.session_state['last_results'] = results
            st.session_state['last_results_ts'] = datetime.now().isoformat()
            
            # (Optional) Clear old summaries if new simulation runs
            # Clear old AI summary since it's now invalid for the new results
            st.session_state.pop('ai_summary', None)
            st.session_state.pop('ai_summary_json', None)
            st.session_state.pop('ai_summary_error', None)
                
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Check your CSV columns. They must be 'Date' and 'Rain'.")

# --- RESULTS DISPLAY (Always runs if results exist) ---
if 'last_results' in st.session_state:
    results = st.session_state['last_results']
    st.header("Step 2: Results")

    # Inject custom CSS to reduce the font size of the metric values.
    # This helps prevent the units from being hidden when the sidebar is open.
    st.markdown("""
        <style>
        [data-testid="stMetricValue"] {
            font-size: 1.75rem;
        }
        </style>
    """, unsafe_allow_html=True)

    total_rain = results['Rain'].sum()
    mean_rain = results['Rain'].mean()
    total_interception = results['i'].sum()
    total_runoff = results['q'].sum()
    mean_soil = results['s'].mean()
    total_et = results['et'].sum()
    total_leakage = results['l'].sum()

    cols = st.columns(7)
    cols[0].metric("Total Rain", f"{total_rain:.1f} mm")
    cols[1].metric("Mean Rain/day", f"{mean_rain:.2f} mm/d")
    cols[2].metric("Total Interception", f"{total_interception:.1f} mm")
    cols[3].metric("Total Runoff", f"{total_runoff:.1f} mm")
    cols[4].metric("Total ET", f"{total_et:.1f} mm")
    cols[5].metric("Total Leakage", f"{total_leakage:.1f} mm")
    cols[6].metric("Mean Soil Moisture", f"{mean_soil:.2f}")

    # Render the AI summary block
    # Note: We use the function defined earlier in your code
    show_ai_summary_block(results, button_key="generate_ai_summary_persistent")

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
