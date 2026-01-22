# Hydrology: Water Balance Model (Streamlit)

This project is a small Streamlit app that runs a soil moisture water balance model (Laio et al., 2001).

Quick start

1. Create a virtualenv and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the app:

```bash
streamlit run app.py
```

Notes / Troubleshooting

- If you want interactive map click/drag support, install the optional packages `folium` and `streamlit-folium` (they are included in `requirements.txt`). If they're not installed the app falls back to `st.map`.
- The app uses the Open-Meteo APIs for geocoding and historical weather. Ensure your machine has network connectivity.
- The TAMU AI integration is optional; if Auto-Tune fails to parse the LLM's response the app will display the raw response so you can inspect it and decide whether a different model or prompt is needed.

Tests

Run the unit tests with:

```bash
python -m unittest discover -v
```

Next enhancements

- Capture drag-end events for markers in the folium map (requires a small JS/Leaflet plugin).
- Add more robust parsing/validation around AI responses, or pin a specific model that returns strict JSON.
