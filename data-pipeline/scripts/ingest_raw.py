import pandas as pd
import numpy as np
import xarray as xr
import h5py
import imdlib as imd
import yaml
import os

def load_config(config_path='../pipeline_config.yaml'):
    if not os.path.exists(config_path):
        # Try parent directory if run from scripts folder
        config_path = os.path.join('..', config_path)
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def process_imd_binary(year, var_type, raw_dir, lat_min, lat_max, lon_min, lon_max):
    """
    Reads IMD binary gridded files using imdlib and crops them to the bounding box.
    var_type must be one of: 'rain', 'tmax', 'tmin'
    """
    # Look for files under <raw_dir>/<var_type>/<year>.grd or similar
    var_folder = os.path.join(raw_dir, var_type)
    if not os.path.exists(var_folder) or not os.listdir(var_folder):
        print(f"No files found in {var_folder}. Skipping IMD {var_type} for {year}.")
        return pd.DataFrame()

    try:
        # imdlib expects raw_dir to contain the <var_type> subfolders.
        # It opens the file from raw_dir/<var_type>/<year>.grd
        data = imd.open_data(var_type, year, year, 'yearwise', raw_dir)
        ds = data.get_xarray()
        
        # Crop to bounding box before converting to DataFrame to save memory
        ds_cropped = ds.sel(lat=slice(lat_min, lat_max), lon=slice(lon_min, lon_max))
        df = ds_cropped.to_dataframe().reset_index()
        
        # Standardize column names
        df.rename(columns={'time': 'datetime'}, inplace=True)
        df['datetime'] = pd.to_datetime(df['datetime']).dt.date
        df['datetime'] = pd.to_datetime(df['datetime'])
        df['source'] = 'imd_ground'
        
        # Drop masked/invalid values (IMD uses -999.0 for oceans/outside India)
        df = df[df[var_type] > -100.0]
        return df
    except Exception as e:
        print(f"Error opening/processing IMD {var_type} for {year}: {e}")
        return pd.DataFrame()

def process_mosdac_h5(file_path, product_key, lat_min, lat_max, lon_min, lon_max):
    """
    Reads MOSDAC INSAT-3D/3DR HDF5 files and filters them by bounding box.
    product_key defines which layer to extract (e.g., 'LST', 'SST', 'IMC')
    """
    print(f"Ingesting MOSDAC {product_key} from {os.path.basename(file_path)}...")
    
    try:
        with h5py.File(file_path, 'r') as h5_data:
            lats = np.array(h5_data['Latitude'])
            lons = np.array(h5_data['Longitude'])
            
            if product_key == 'LST':
                layer = np.array(h5_data['Geophysical_Data']['LST'])
            elif product_key == 'SST':
                layer = np.array(h5_data['Geophysical_Data']['SST'])
            elif product_key == 'IMC':
                layer = np.array(h5_data['Geophysical_Data']['Rainfall'])
            else:
                raise ValueError("Unknown MOSDAC product key")
                
            acq_time = h5_data.attrs.get('Acquisition_Time', '2021-06-27T00:00:00Z')
            if hasattr(acq_time, 'decode'):
                acq_time = acq_time.decode('utf-8')
            elif isinstance(acq_time, bytes):
                acq_time = acq_time.decode('utf-8')
    except Exception as e:
        print(f"Error reading HDF5 structure from {file_path}: {e}")
        return pd.DataFrame()

    # Flatten coordinates and values
    flat_lats = lats.flatten()
    flat_lons = lons.flatten()
    flat_vals = layer.flatten()
    
    # Filter using bounding box immediately to save memory
    mask = (flat_lats >= lat_min) & (flat_lats <= lat_max) & (flat_lons >= lon_min) & (flat_lons <= lon_max)
    
    df = pd.DataFrame({
        'lat': flat_lats[mask],
        'lon': flat_lons[mask],
        product_key.lower(): flat_vals[mask],
        'datetime': pd.to_datetime(acq_time)
    })
    
    # Strip timezone to ensure consistency when merging with ground IMD data
    if df['datetime'].dt.tz is not None:
        df['datetime'] = df['datetime'].dt.tz_localize(None)
        
    # Normalize time to midnight (00:00:00) to match daily frequency and avoid duplicate rows
    df['datetime'] = df['datetime'].dt.date
    df['datetime'] = pd.to_datetime(df['datetime'])
        
    df['source'] = 'insat_satellite'
    
    # Filter out nodata/fill values
    df = df.dropna()
    df = df[df[product_key.lower()] < 1000] # Standard threshold
    
    return df

def build_master_grid(config):
    """
    Orchestrates the ingestion of ground and satellite datasets for target years and pilot bounds.
    Returns a dictionary of DataFrames grouped by variable.
    """
    # 1. Load spatial and temporal configurations
    spatial = config['spatial_settings']
    lat_min, lat_max = spatial['lat_min'], spatial['lat_max']
    lon_min, lon_max = spatial['lon_min'], spatial['lon_max']
    
    temporal = config['temporal_settings']
    start_year = temporal['start_year']
    end_year = temporal['end_year']
    
    years = list(range(start_year, end_year + 1))
    print(f"Processing data for bounding box: Lat [{lat_min}, {lat_max}], Lon [{lon_min}, {lon_max}] for years {years}")

    datasets = {
        'rain': [],
        'tmax': [],
        'tmin': [],
        'lst': [],
        'sst': [],
        'imc': []
    }
    
    # 2. Process IMD Ground Data for all years
    imd_dir = config['paths']['raw_imd_dir']
    if os.path.exists(imd_dir):
        for year in years:
            for var_type in ['rain', 'tmax', 'tmin']:
                df_var = process_imd_binary(year, var_type, imd_dir, lat_min, lat_max, lon_min, lon_max)
                if not df_var.empty:
                    datasets[var_type].append(df_var)
    
    # 3. Process MOSDAC Satellite Data
    mosdac_dir = config['paths']['raw_mosdac_dir']
    if os.path.exists(mosdac_dir):
        for file in os.listdir(mosdac_dir):
            if file.endswith('.h5'):
                file_path = os.path.join(mosdac_dir, file)
                
                if 'LST' in file:
                    df = process_mosdac_h5(file_path, 'LST', lat_min, lat_max, lon_min, lon_max)
                    if not df.empty:
                        datasets['lst'].append(df)
                elif 'SST' in file:
                    df = process_mosdac_h5(file_path, 'SST', lat_min, lat_max, lon_min, lon_max)
                    if not df.empty:
                        datasets['sst'].append(df)
                elif 'IMC' in file:
                    df = process_mosdac_h5(file_path, 'IMC', lat_min, lat_max, lon_min, lon_max)
                    if not df.empty:
                        datasets['imc'].append(df)

    # 4. Consolidate list of DataFrames to single DataFrames per variable
    consolidated_datasets = {}
    for var, dfs in datasets.items():
        if dfs:
            print(f"Consolidating datasets for variable: {var}...")
            consolidated_datasets[var] = pd.concat(dfs, ignore_index=True)
            print(f"Variable '{var}' shape: {consolidated_datasets[var].shape}")
            
    return consolidated_datasets

if __name__ == "__main__":
    cfg = load_config('pipeline-config.yaml')
    datasets = build_master_grid(cfg)
    for var, df in datasets.items():
        print(f"Processed {var} data: {df.shape[0]} rows.")