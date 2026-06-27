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

def regrid_to_target(df, target_resolution=0.25):
    """
    Standardizes different coordinate systems into a unified grid.
    Uses nearest-neighbor interpolation to target resolution.
    """
    print(f"Regridding data to {target_resolution} degree resolution...")
    
    if df.empty:
        return df

    # Create target grid
    lat_min, lat_max = df['lat'].min(), df['lat'].max()
    lon_min, lon_max = df['lon'].min(), df['lon'].max()
    
    grid_lat = np.arange(lat_min, lat_max, target_resolution)
    grid_lon = np.arange(lon_min, lon_max, target_resolution)
    grid_lon, grid_lat = np.meshgrid(grid_lon, grid_lat)
    
    # Identify value columns (excluding lat, lon, datetime, source)
    value_cols = [c for c in df.columns if c not in ['lat', 'lon', 'datetime', 'source']]
    
    regridded_dfs = []
    # Process each timestamp separately
    for ts in df['datetime'].unique():
        ts_df = df[df['datetime'] == ts]
        
        points = ts_df[['lon', 'lat']].values
        
        ts_regridded = pd.DataFrame({
            'lon': grid_lon.flatten(),
            'lat': grid_lat.flatten(),
            'datetime': ts
        })
        
        for col in value_cols:
            values = ts_df[col].values
            grid_values = griddata(points, values, (grid_lon, grid_lat), method='nearest')
            ts_regridded[col] = grid_values.flatten()
            
        regridded_dfs.append(ts_regridded)
        
    return pd.concat(regridded_dfs, ignore_index=True)

def convert_to_spatial_dataframe(df):
    """Converts standard lat/lon rows into PostGIS geometry strings."""
    print("Converting coordinates to PostGIS geometry points...")
    # SRID 4326 represents standard WGS 84 GPS coordinates
    df['geom'] = df.apply(lambda row: f"SRID=4326;POINT({row['lon']} {row['lat']})", axis=1)
    return df

def main():
    # 1. Load Configurations
    config = load_config('pipeline-config.yaml')
    
    # 2. Run Ingestion (Combines IMD and MOSDAC)
    raw_master_df = build_master_grid(config)
    if raw_master_df.empty:
        print("Workflow stopped: No data ingested.")
        return

    # 3. Align & Regrid Coordinates
    aligned_df = regrid_to_target(raw_master_df, config['grid_settings']['resolution_deg'])

    # 4. Clean Gaps & Detect Anomalies
    z_limit = config.get('thresholds', {}).get('temp_zscore_limit', 3.0)
    cleaned_df = clean_pipeline(aligned_df, z_limit)

    # 5. Extract Digital Twin Features
    processed_df = run_feature_engineering(cleaned_df)

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