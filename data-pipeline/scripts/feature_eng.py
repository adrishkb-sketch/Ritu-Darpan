# scripts/feature_eng.py
import pandas as pd
import numpy as np

def run_feature_engineering(df):
    """
    Extracts digital twin features including rolling metrics and lags.
    """
    print("Running feature engineering...")
    
    # Ensure data is sorted by lat, lon, and datetime for correct window calculations
    df = df.sort_values(by=['lat', 'lon', 'datetime'])
    
    # Group by spatial grid point
    grouped = df.groupby(['lat', 'lon'])
    
    # 1. Rolling Metrics (7-day window)
    if 'rain' in df.columns:
        df['rain_7d_avg'] = grouped['rain'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        df['rain_lag_1d'] = grouped['rain'].shift(1)
        
    if 'tmax' in df.columns:
        df['tmax_7d_avg'] = grouped['tmax'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())
        df['temp_volatility'] = grouped['tmax'].transform(lambda x: x.rolling(window=7, min_periods=1).std())
        
    if 'lst' in df.columns:
        df['lst_7d_avg'] = grouped['lst'].transform(lambda x: x.rolling(window=7, min_periods=1).mean())

    # 2. Infrastructure/Contextual Features (Dummy examples for Digital Twin)
    # In a real scenario, these might come from GIS layers
    df['is_heatwave_risk'] = (df['tmax'] > 40) if 'tmax' in df.columns else False
    
    # Fill any NaNs created by lagging
    df = df.fillna(0)
    
    return df
