import numpy as np
import pandas as pd

def water_balance_simulation(p, rain_df):
    """
    Runs the soil water balance simulation.
    
    Parameters:
    p (dict): Dictionary of soil parameters (sh, sw, sstar, etc.)
    rain_df (pd.DataFrame): DataFrame with a 'Rain' column (daily rainfall in mm)
    
    Returns:
    pd.DataFrame: The original data with added columns for s (soil moisture), q (runoff), etc.
    """
    
    # 1. Setup outputs
    days = len(rain_df)
    # Initialize arrays to store results
    s_out = np.zeros(days)  # Soil moisture
    q_out = np.zeros(days)  # Runoff
    et_out = np.zeros(days) # Evapotranspiration
    l_out = np.zeros(days)  # Leakage
    i_out = np.zeros(days)  # Interception
    
    # 2. Set initial soil moisture (s0)
    # If not provided in p, assume 30% saturation
    s = p.get('s0', 0.3) 
    
    # 3. Time Loop
    for t in range(days):
        R = rain_df['Rain'].iloc[t] # Rainfall for today
        
        # --- The Math (from your Notebook) ---
        
        # Interception (I)
        # Simplified: Assume some rain is caught by leaves before hitting soil
        # (You can adjust this logic based on the notebook's specific formula)
        I = min(R, 0.2) # Example: Max 0.2mm interception
        
        # Infiltration
        inf = R - I
        
        # Evapotranspiration (ET)
        # Piecewise function based on soil moisture 's'
        if s <= p['sh']:
            ET = 0
        elif s <= p['sw']:
            ET = p['ew'] * (s - p['sh']) / (p['sw'] - p['sh'])
        elif s <= p['sstar']:
            ET = p['ew'] + (p['emax'] - p['ew']) * (s - p['sw']) / (p['sstar'] - p['sw'])
        else:
            ET = p['emax']
            
        # Leakage (L)
        if s <= p['sfc']:
            L = 0
        else:
            L = p['ks'] * (np.exp(p['beta'] * (s - p['sfc'])) - 1) / (np.exp(p['beta'] * (1 - p['sfc'])) - 1)
            
        # Runoff (Q) mechanism (Saturation excess)
        # Calculate new storage
        # n*Zr is the water holding capacity depth
        storage_capacity = p['n'] * p['zr']
        
        # Water Balance Equation: ds/dt = (Infiltration - ET - L - Q) / (n*Zr)
        # We solve strictly:
        
        # Potential new saturation
        s_next = s + (inf - ET - L) / storage_capacity
        
        Q = 0
        if s_next > 1.0:
            # Soil is full, excess becomes Runoff
            excess_s = s_next - 1.0
            Q = excess_s * storage_capacity
            s_next = 1.0
        elif s_next < 0:
            s_next = 0
            
        # Store results for this day
        s_out[t] = s_next
        q_out[t] = Q
        et_out[t] = ET
        l_out[t] = L
        i_out[t] = I
        
        # Update s for the next day
        s = s_next

    # 4. Package results
    results = rain_df.copy()
    results['s'] = s_out
    results['q'] = q_out
    results['et'] = et_out
    results['l'] = l_out
    results['i'] = i_out
    
    return results