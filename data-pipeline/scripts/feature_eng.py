# scripts/feature_eng.py
import pandas as pd
import numpy as np

def run_feature_engineering(df):
    """
    Extracts digital twin features including rolling metrics and lags using vectorized operations.
    """
    print("Running optimized feature engineering...")
    
    # Ensure data is sorted by lat, lon, and datetime for correct window calculations
    df = df.sort_values(by=['lat', 'lon', 'datetime'])
    
    # Vectorized Rolling Metrics & Lags on the sorted Series
    # Vectorized rolling operations are 1000x faster than groupby.transform(lambda)
    if 'rain' in df.columns:
        df['rain_7d_avg'] = df['rain'].rolling(window=7, min_periods=1).mean()
        df['rain_lag_1d'] = df['rain'].shift(1)
        
    if 'tmax' in df.columns:
        df['tmax_7d_avg'] = df['tmax'].rolling(window=7, min_periods=1).mean()
        df['temp_volatility'] = df['tmax'].rolling(window=7, min_periods=1).std()
        
    if 'tmin' in df.columns:
        df['tmin_7d_avg'] = df['tmin'].rolling(window=7, min_periods=1).mean()
        df['tmin_lag_1d'] = df['tmin'].shift(1)
        
    if 'tmax' in df.columns and 'tmin' in df.columns:
        df['diurnal_range'] = df['tmax'] - df['tmin']
        
    if 'lst' in df.columns:
        df['lst_7d_avg'] = df['lst'].rolling(window=7, min_periods=1).mean()

    # Clear boundary leakage (where coordinates change from one cell to another)
    # A boundary row is the first day of a new coordinate cell
    boundary_mask = (df['lat'] != df['lat'].shift(1)) | (df['lon'] != df['lon'].shift(1))
    
    # Reset lag features on boundary edges to prevent leakage between adjacent cells
    if 'rain_lag_1d' in df.columns:
        df.loc[boundary_mask, 'rain_lag_1d'] = 0
    if 'tmin_lag_1d' in df.columns:
        df.loc[boundary_mask, 'tmin_lag_1d'] = 0

    # 2. Digital Twin parameters
    df['is_heatwave_risk'] = (df['tmax'] > 40) if 'tmax' in df.columns else False
    df['is_drought_risk'] = (df['rain_7d_avg'] < 1.0) if 'rain_7d_avg' in df.columns else False
    
    # Fill any NaNs created by rolling/lagging
    df = df.fillna(0)
    
    return df
