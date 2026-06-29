# data_pipeline_main.py
import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
try:
    from geoalchemy2 import Geometry, WKTElement
except ImportError:
    Geometry = None
    WKTElement = None
import yaml
from scipy.interpolate import griddata
import json
from shapely.geometry import shape, Point
from shapely.ops import unary_union
from shapely.prepared import prep

# Import functions from scripts module
from scripts.ingest_raw import load_config, build_master_grid
from scripts.clean_anomalies import clean_pipeline
from scripts.feature_eng import run_feature_engineering

def connect_to_db(config):
    """Establishes database connection using config."""
    db_cfg = config.get('database', {})
    user = db_cfg.get('user', 'username')
    password = db_cfg.get('password', 'password')
    host = db_cfg.get('host', 'localhost')
    port = db_cfg.get('port', '5432')
    dbname = db_cfg.get('dbname', 'climate_twin_db')
    
    DB_URL = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
    return create_engine(DB_URL)

def regrid_variable(var_name, var_df, target_grid_coords, method='linear'):
    """
    Interpolates a single variable's DataFrame to the target grid.
    Performs interpolation day-by-day to maintain temporal index.
    """
    print(f"Regridding variable '{var_name}' to target grid coordinates...")
    if var_df.empty:
        return pd.DataFrame()
        
    regridded_days = []
    grid_points = target_grid_coords[['lon', 'lat']].values
    
    # Group by datetime to avoid O(N) scanning inside the loop
    for dt, day_df in var_df.groupby('datetime'):
        points = day_df[['lon', 'lat']].values
        values = day_df[var_name].values
        
        # Check if we have enough points to interpolate
        if len(points) < 4:
            day_method = 'nearest'
        else:
            day_method = method
            
        # Interpolate
        grid_vals = griddata(points, values, grid_points, method=day_method)
        
        # Fill edge boundary NaNs with nearest neighbor
        if np.isnan(grid_vals).any():
            nan_mask = np.isnan(grid_vals)
            grid_vals_nearest = griddata(points, values, grid_points[nan_mask], method='nearest')
            grid_vals[nan_mask] = grid_vals_nearest
            
        day_regridded = target_grid_coords.copy()
        day_regridded['datetime'] = dt
        day_regridded[var_name] = grid_vals
        
        regridded_days.append(day_regridded)
        
    if not regridded_days:
        return pd.DataFrame()
        
    return pd.concat(regridded_days, ignore_index=True)

def merge_regridded_datasets(datasets, config):
    """
    Regrids each variable separately and then horizontally merges them on datetime, lat, lon.
    """
    spatial = config['spatial_settings']
    lat_min, lat_max = spatial['lat_min'], spatial['lat_max']
    lon_min, lon_max = spatial['lon_min'], spatial['lon_max']
    resolution = config['grid_settings']['resolution_deg']
    
    # Establish target grid coordinate lists
    target_lats = np.arange(lat_min, lat_max, resolution)
    target_lons = np.arange(lon_min, lon_max, resolution)
    grid_lon, grid_lat = np.meshgrid(target_lons, target_lats)
    
    target_grid_coords = pd.DataFrame({
        'lon': grid_lon.flatten(),
        'lat': grid_lat.flatten()
    })
    
    merged_df = None
    
    for var_name, var_df in datasets.items():
        if var_df.empty:
            continue
        
        # Regrid this variable
        var_regridded = regrid_variable(var_name, var_df, target_grid_coords, method='linear')
        if var_regridded.empty:
            continue
            
        if merged_df is None:
            merged_df = var_regridded
        else:
            # Merge horizontally
            merged_df = pd.merge(merged_df, var_regridded, on=['datetime', 'lat', 'lon'], how='outer')
            
    if merged_df is not None:
        merged_df['source'] = 'fused_twin'
        
    return merged_df

def convert_to_spatial_dataframe(df):
    """Converts standard lat/lon rows into PostGIS geometry strings using vectorized operations."""
    print("Converting coordinates to PostGIS geometry points...")
    # Vectorized string concatenation is 100x faster than df.apply(lambda)
    df['geom'] = "SRID=4326;POINT(" + df['lon'].astype(str) + " " + df['lat'].astype(str) + ")"
    return df

def main():
    # 1. Load Configurations
    config = load_config('pipeline-config.yaml')
    
    # 2. Run Ingestion (Gets dictionary of DataFrames per variable)
    datasets = build_master_grid(config)
    if not datasets:
        print("Workflow stopped: No data ingested.")
        return
 
    # 3. Align & Regrid Coordinates separately and horizontally merge
    print("Regridding and fusing variables horizontally...")
    fused_df = merge_regridded_datasets(datasets, config)
    if fused_df is None or fused_df.empty:
        print("Workflow stopped: Regridding failed.")
        return
 
    # 4. Clean Gaps & Detect Anomalies
    z_limit = config.get('thresholds', {}).get('temp_zscore_limit', 3.0)
    cleaned_df = clean_pipeline(fused_df, z_limit)
 
    # 5. Extract Digital Twin Features
    processed_df = run_feature_engineering(cleaned_df, config)

    # 6. Format Data for PostGIS Spatial Database
    spatial_df = convert_to_spatial_dataframe(processed_df)

    # 7. Write to PostgreSQL or CSV
    try:
        db_engine = connect_to_db(config)
        print("Uploading synchronized climate layers to PostGIS...")
        if Geometry is not None:
             spatial_df.to_sql(
                name='dynamic_climate_grid',
                con=db_engine,
                if_exists='append',
                index=False,
                chunksize=5000,
                dtype={'geom': Geometry(geometry_type='POINT', srid=4326)}
            )
        else:
            print("GeoAlchemy2 not found. Skipping PostGIS upload.")
            raise Exception("GeoAlchemy2 missing")
            
    except Exception as e:
        print(f"PostGIS upload failed or skipped: {e}")
        output_dir = config['paths']['processed_data_dir']
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, config['paths']['output_file'])
        spatial_df.to_csv(output_path, index=False)
        print(f"Data saved to CSV instead: {output_path}")

    print("🎉 Pipeline run successful! Digital Twin data synchronized.")

if __name__ == "__main__":
    main()