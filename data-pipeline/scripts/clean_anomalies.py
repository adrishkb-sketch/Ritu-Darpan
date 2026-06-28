# scripts/clean_anomalies.py
import pandas as pd
import numpy as np
from scipy import stats

def clean_pipeline(df, z_threshold=3):
    """
    Detects anomalies using Z-score and fills gaps using linear interpolation.
    """
    print(f"Running cleaning pipeline with Z-score threshold: {z_threshold}...")
    
    # Ensure data is sorted by coordinates and time for proper temporal interpolation
    df = df.sort_values(by=['lat', 'lon', 'datetime'])
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        # Skip lat/lon as they are part of the grid structure
        if col in ['lat', 'lon']:
            continue
            
        print(f"Cleaning column: {col}...")
        
        # 1. Anomaly Detection
        # Calculate Z-scores
        z_scores = np.abs(stats.zscore(df[col], nan_policy='omit'))
        
        # Replace anomalies with NaN
        df.loc[z_scores > z_threshold, col] = np.nan
        
        # 2. Gap Filling (Fast vectorized forward/backward fill temporally for each grid point)
        df[col] = df.groupby(['lat', 'lon'])[col].ffill().bfill()
        
        # Fill remaining NaNs if any (e.g., at edges) with mean
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())
            
    return df
