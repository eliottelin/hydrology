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

# Create a default dummy dataset (fallback)
dates = pd.date_range(start="2024-01-01", periods=100)
np.random.seed(42)
rain_vals = np.random.exponential(scale=5, size=100) * (np.random.rand(100) > 0.7)
default_df = pd.DataFrame({'Date': dates, 'Rain': rain_vals})

data_source = st.radio("Choose Data Source:", ["Use Example Data", "Upload CSV"])

rain_df = None

if data_source == "Upload CSV":
    uploaded_file = st.file_uploader("Upload CSV (NOAA Format OK)")
    if uploaded_file:
        # Read the raw file
        raw_df = pd.read_csv(uploaded_file)
        
        # 1. CLEANING: Standardize Column Names
        # Make a copy to avoid SettingWithCopy warnings
        df_clean = raw_df.copy()
        
        # Renaissance of columns: Map NOAA names to our names
        col_map = {'DATE': 'Date', 'PRCP': 'Rain', 'Prcp': 'Rain', 'Rainfall': 'Rain'}
        df_clean = df_clean.rename(columns=col_map)
        
        # 2. FILTERING: Handle Multiple Stations
        # If the file has a 'NAME' or 'STATION' column, let the user pick one.
        if 'NAME' in df_clean.columns:
            stations = df_clean['NAME'].unique()
            if len(stations) > 1:
                st.warning(f"⚠️ Multiple stations detected ({len(stations)} found).")
                selected_station = st.selectbox("Select a Station to Analyze:", stations)
                # Filter data to just that station
                df_clean = df_clean[df_clean['NAME'] == selected_station]
                st.success(f"loaded data for: {selected_station}")
        
        elif 'STATION' in df_clean.columns:
             stations = df_clean['STATION'].unique()
             if len(stations) > 1:
                selected_station = st.selectbox("Select a Station ID:", stations)
                df_clean = df_clean[df_clean['STATION'] == selected_station]

        # 3. UNIT CONVERSION
        # NOAA data is usually in Inches. The model needs Millimeters.
        # We assume if the max rain is small (< 15), it might be inches.
        if 'Rain' in df_clean.columns:
            # Check for NaN values and fill them (common in NOAA data)
            df_clean['Rain'] = df_clean['Rain'].fillna(0)
            
            is_inches = st.checkbox("Convert Inches to MM? (Check this for NOAA data)", value=True)
            if is_inches:
                df_clean['Rain'] = df_clean['Rain'] * 25.4
                st.caption("Converted values x 25.4 (Inches -> MM)")
            
            # Final check to ensure we have the columns we need
            rain_df = df_clean[['Date', 'Rain']].sort_values('Date')
        else:
            st.error("Could not find a 'PRCP' or 'Rain' column in your CSV.")
            
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
            
            labels_map = {
                'Date': 'Date',
                's': 'Soil Moisture (0-1)',
                'q': 'Runoff (mm/day)',
                'et': 'Evapotranspiration (mm/day)',
                'l': 'Leakage (mm/day)',
                'i': 'Canopy Interception (mm)',  # New Label
                'Rain': 'Precipitation (mm)'      # New Label
            }
            
            # 1. RAIN & INTERCEPTION (The Inputs)
            st.subheader("Inputs: Precipitation & Interception")
            
            # Create a combined bar chart for Rain vs Interception
            # We melt the dataframe to plot two variables on one bar chart
            input_df = results[['Date', 'Rain', 'i']].melt('Date', var_name='Variable', value_name='Amount (mm)')
            input_df['Variable'] = input_df['Variable'].map({'Rain': 'Total Rain', 'i': 'Interception Loss'})
            
            fig_rain = px.bar(
                input_df, 
                x='Date', 
                y='Amount (mm)', 
                color='Variable', 
                barmode='overlay', # Overlays them so you see how much rain was "eaten" by interception
                title="Rainfall vs. Canopy Interception",
                opacity=0.7
            )
            st.plotly_chart(fig_rain, use_container_width=True)

            # 2. SOIL MOISTURE
            st.subheader("State: Soil Moisture")
            fig_s = px.line(
                results, x='Date', y='s', 
                title="Soil Moisture Dynamics", 
                range_y=[0,1],
                labels=labels_map
            )
            fig_s.add_hline(y=p['sw'], line_dash="dash", line_color="red", annotation_text="Wilting Point")
            fig_s.add_hline(y=p['sfc'], line_dash="dash", line_color="green", annotation_text="Field Capacity")
            st.plotly_chart(fig_s, use_container_width=True)
            
            # 3. OUTPUTS (Runoff, ET, Leakage)
            st.subheader("Outputs: Runoff, ET, Leakage")
            
            # Runoff
            fig_q = px.bar(results, x='Date', y='q', title="Runoff (Q)", labels=labels_map)
            fig_q.update_yaxes(title_text="Runoff (mm)")
            st.plotly_chart(fig_q, use_container_width=True)

            # ET and Leakage (Grouped together as losses)
            loss_df = results[['Date', 'et', 'l']].melt('Date', var_name='Variable', value_name='Amount (mm)')
            loss_df['Variable'] = loss_df['Variable'].map({'et': 'Evapotranspiration', 'l': 'Leakage'})
            
            fig_loss = px.line(
                loss_df, 
                x='Date', 
                y='Amount (mm)', 
                color='Variable',
                title="Losses: Evapotranspiration & Leakage"
            )
            st.plotly_chart(fig_loss, use_container_width=True)
            
            # 3. Data Table
            with st.expander("View Raw Data"):
                st.dataframe(results)
                
            # 4. Download
            csv = results.to_csv(index=False).encode('utf-8')
            st.download_button("Download Results", csv, "simulation_results.csv", "text/csv")
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.write("Check your CSV columns. They must be 'Date' and 'Rain'.")