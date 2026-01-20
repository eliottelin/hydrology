import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from model import water_balance_simulation

# --- Page Config ---
st.set_page_config(page_title="Hydrology Lab Model", layout="wide")

st.title("🌊 Hydrology Lab: Water Balance Simulation")
st.markdown("This app simulates soil moisture dynamics based on the **Laio et al. (2001)** model.")

# --- SIDEBAR: PARAMETERS ---
st.sidebar.header("1. Soil Properties")

# Dictionary to store parameters
p = {}

# Sliders for each variable (Defaults taken from your Notebook)
p['sh'] = st.sidebar.number_input("Hygroscopic Point (sh)", 0.0, 1.0, 0.08, format="%.2f")
p['sw'] = st.sidebar.number_input("Wilting Point (sw)", 0.0, 1.0, 0.11, format="%.2f")
p['sstar'] = st.sidebar.number_input("Stomatal Closure (s*)", 0.0, 1.0, 0.33, format="%.2f")
p['sfc'] = st.sidebar.number_input("Field Capacity (sfc)", 0.0, 1.0, 0.40, format="%.2f")
p['n'] = st.sidebar.number_input("Porosity (n)", 0.1, 1.0, 0.55, format="%.2f")
p['zr'] = st.sidebar.number_input("Root Depth (Zr) [mm]", 10.0, 2000.0, 500.0) # Changed default to 500mm (50cm) to be realistic
p['ks'] = st.sidebar.number_input("Saturated Conductivity (Ks) [mm/day]", 0.0, 1000.0, 200.0)

st.sidebar.header("2. Vegetation")
p['ew'] = st.sidebar.number_input("Evap at Wilting (Ew) [mm/day]", 0.0, 10.0, 0.1)
p['emax'] = st.sidebar.number_input("Max Evap (Emax) [mm/day]", 0.0, 20.0, 5.0)
p['beta'] = st.sidebar.number_input("Leakage Parameter (Beta)", 0.0, 20.0, 3.0)

# Initial Condition
p['s0'] = st.sidebar.slider("Initial Soil Moisture (s0)", 0.0, 1.0, 0.3)

# --- WEATHER DATA ---
st.header("Step 1: Weather Data")

# Create a default dummy dataset (so the app isn't empty)
dates = pd.date_range(start="2024-01-01", periods=100)
# Create synthetic rain: mostly zeros, some random storms
np.random.seed(42)
rain_vals = np.random.exponential(scale=5, size=100) * (np.random.rand(100) > 0.7)
default_df = pd.DataFrame({'Date': dates, 'Rain': rain_vals})

data_source = st.radio("Choose Data Source:", ["Use Example Data", "Upload CSV"])

rain_df = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV (Columns: 'Date', 'Rain')")
    if uploaded_file:
        rain_df = pd.read_csv(uploaded_file)
    
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
else:
    rain_df = default_df
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
            
            # 1. Soil Moisture Plot
            fig_s = px.line(results, x='Date', y='s', title="Soil Moisture (s) Over Time", range_y=[0,1])
            # Add horizontal lines for critical points
            fig_s.add_hline(y=p['sw'], line_dash="dash", line_color="red", annotation_text="Wilting Point")
            fig_s.add_hline(y=p['sfc'], line_dash="dash", line_color="green", annotation_text="Field Capacity")
            st.plotly_chart(fig_s, use_container_width=True)
            
            # 2. Runoff Plot
            fig_q = px.bar(results, x='Date', y='q', title="Runoff (Q) Events")
            st.plotly_chart(fig_q, use_container_width=True)

            # 3. Evapotranspiration (ET) Plot
            fig_et = px.line(results, x='Date', y='et', title="Evapotranspiration (ET) Over Time")
            st.plotly_chart(fig_et, use_container_width=True)

            # 4. Leakage (L) Plot
            fig_l = px.line(results, x='Date', y='l', title="Deep Drainage / Leakage (L)")
            st.plotly_chart(fig_l, use_container_width=True)
            
            # 5. Data Table
            with st.expander("View Raw Data"):
                st.dataframe(results)
                
            # 6. Download
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "simulation_results.csv", "text/csv")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Check your CSV columns. They must be 'Date' and 'Rain'.")