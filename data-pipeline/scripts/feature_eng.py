# scripts/feature_eng.py
import pandas as pd
import numpy as np
import os
import json
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from scipy.ndimage import uniform_filter

def run_feature_engineering(df, config=None):
    """
    Extracts digital twin features including elevation, spatial gradients, rolling metrics, and lags.
    Applies West Bengal shapefile boundary masking at the end to ensure correct spatial features.
    """
    print("Running advanced spatially aware feature engineering...")
    
    # 1. Topographic Elevation Model
    print("Computing topographic elevation covariate...")
    lats = df['lat'].values
    lons = df['lon'].values
    elevation = np.where(lats > 26.5, (lats - 26.5) * 3000 + 150, (lats - 21.0) * 20)
    # Add Purulia/Western plateau heights
    elevation += np.where(lons < 87.0, (87.0 - lons) * 150, 0)
    df['elevation'] = np.clip(elevation, 0, 3600)
    
    # 2. Spatially Aware Features (computed on the full rectangular grid)
    print("Computing spatial gradients and neighborhood transport averages...")
    df = df.sort_values(by=['datetime', 'lat', 'lon'])
    
    num_days = df['datetime'].nunique()
    num_lats = df['lat'].nunique()
    num_lons = df['lon'].nunique()
    total_points = len(df)
    
    if total_points == num_days * num_lats * num_lons:
        tmax_3d = df['tmax'].values.reshape(num_days, num_lats, num_lons)
        rain_3d = df['rain'].values.reshape(num_days, num_lats, num_lons)
        
        # Spatial gradients
        tmax_grad_y, tmax_grad_x = np.gradient(tmax_3d, axis=(1, 2))
        rain_grad_y, rain_grad_x = np.gradient(rain_3d, axis=(1, 2))
        
        # Neighborhood spatial means (3x3 grid filter)
        tmax_smooth = uniform_filter(tmax_3d, size=(1, 3, 3), mode='nearest')
        rain_smooth = uniform_filter(rain_3d, size=(1, 3, 3), mode='nearest')
        
        df['tmax_grad_x'] = tmax_grad_x.flatten()
        df['tmax_grad_y'] = tmax_grad_y.flatten()
        df['rain_grad_x'] = rain_grad_x.flatten()
        df['rain_grad_y'] = rain_grad_y.flatten()
        df['tmax_spatial_mean'] = tmax_smooth.flatten()
        df['rain_spatial_mean'] = rain_smooth.flatten()
    else:
        print("Warning: Grid shapes are irregular. Skipping 3D spatial reshaping.")
        df['tmax_grad_x'] = 0.0
        df['tmax_grad_y'] = 0.0
        df['rain_grad_x'] = 0.0
        df['rain_grad_y'] = 0.0
        df['tmax_spatial_mean'] = df['tmax']
        df['rain_spatial_mean'] = df['rain']
        
    # Ensure data is sorted by lat, lon, and datetime for lag calculations
    df = df.sort_values(by=['lat', 'lon', 'datetime'])
    
    # Vectorized Rolling Metrics & Lags
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

    # Reset boundary leakages
    boundary_mask = (df['lat'] != df['lat'].shift(1)) | (df['lon'] != df['lon'].shift(1))
    if 'rain_lag_1d' in df.columns:
        df.loc[boundary_mask, 'rain_lag_1d'] = 0
    if 'tmin_lag_1d' in df.columns:
        df.loc[boundary_mask, 'tmin_lag_1d'] = 0

    # Digital Twin risk metrics
    df['is_heatwave_risk'] = (df['tmax'] > 40) if 'tmax' in df.columns else False
    df['is_drought_risk'] = (df['rain_7d_avg'] < 1.0) if 'rain_7d_avg' in df.columns else False
    
    df = df.fillna(0)
    
    # 4. Filter by West Bengal GeoJSON shape at the very end
    script_dir = os.path.dirname(os.path.abspath(__file__))
    geojson_path = os.path.abspath(os.path.join(script_dir, '..', 'west-bengal.geojson'))
    if os.path.exists(geojson_path):
        print(f"Applying West Bengal state boundary mask from: {geojson_path}")
        with open(geojson_path, 'r') as f:
            geojson_data = json.load(f)
        geoms = [shape(feature["geometry"]) for feature in geojson_data.get("features", [])]
        state_geom = unary_union(geoms)
        prepared_geom = prep(state_geom)
        
        mask = [prepared_geom.contains(Point(lon, lat)) for lon, lat in zip(df['lon'], df['lat'])]
        df = df[mask].reset_index(drop=True)
        print(f"Masked final features to {len(df)} points inside West Bengal.")
    else:
        print(f"WARNING: West Bengal GeoJSON not found at {geojson_path}. No boundary mask applied.")
    
    return df
